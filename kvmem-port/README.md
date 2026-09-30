# kvmem port for prismml llama.cpp

Builds `llama-kvmem-cli` and `llama-kvmem-server` from the
[kvmem-llama.cpp](https://github.com/bellkin/kvmem-llama.cpp) adapter against
this llama.cpp tree. The tree must carry the kvmem patches (the
`kvmem-cuda-port` branch does).

## Layout

This directory can live in two places:

- in-repo: `<llama.cpp checkout>/kvmem-port/` (this is how it is committed)
- standalone: any directory with the llama.cpp checkout next to it as `llama.cpp/`

`CMakeLists.txt` auto-detects which one it is.

## Prerequisites

- CUDA toolkit (12.x) and cmake >= 3.21
- a checkout of kvmem-llama.cpp (defaults to `/home/nbuser/ai/llama/kvmem/kvmem-llama.cpp`;
  override with `-DKVMEM_ROOT=`)

## Build

```sh
cmake -S kvmem-port -B build -DKVMEM_ROOT=/path/to/kvmem-llama.cpp
cmake --build build -j --target llama-kvmem-cli llama-kvmem-server
```

Binaries land in `build/bin/`.

- `CMAKE_CUDA_ARCHITECTURES` defaults to `120` (RTX 5070 / Blackwell); override
  for other GPUs, e.g. `-DCMAKE_CUDA_ARCHITECTURES="80;90"` or `"75"` for Turing.

## Architecture support

Turing (sm_75) and newer are supported. The full `llama` library, including the
kvmem stage-in kernels and the Gated Delta Net kernels, compiles for sm_75.
GDN uses a conservative `cols_per_warp=1` config below Ampere, and flash
attention picks the vector kernel for quantized-KV decode on Turing (prefill
with quantized KV dequantizes to f16 for the MMA path), so functionality is
preserved with some performance cost on older cards.
- `GGML_CUDA_FA_QUANTS=all` is forced on (quantized-KV flash attention used by
  the kvmem path).

## Run

### CLI

```sh
./build/bin/llama-kvmem-cli -m model.gguf --device CUDA0 -ngl 12 -c 2048 "prompt"
```

MTP speculative decoding takes a sidecar GGUF:

```sh
... --spec-type draft-mtp --spec-draft-model /path/to/mtp-Q4_0.gguf --spec-draft-n-max 3
```

Note: the bonsai PTQ1_0 GGUF has no welded nextn layer, so `--spec-draft-model`
is required there. The unsloth Q4_K GGUF has one and works without the flag.

### Server

```sh
./build/bin/llama-kvmem-server -m model.gguf --device CUDA0 -ngl 12 \
    -c 36864 --kvmem-budget 2048 --port 8080
```

- OpenAI-compatible endpoints: `/v1/chat/completions`, `/health`, `/props`, ...
- Requests are gated by `prompt + max_tokens <= n_ctx`, so long-document
  retrieval needs a large `-c` together with a small `--kvmem-budget` (the
  budget sizes the GPU working set; eviction + retrieval happen when the
  prompt exceeds it). A budget that is too small (e.g. 1024) can stage the
  right block yet still degrade answer quality.
- Server MTP: `--spec-type draft-mtp`, with `--spec-draft-model /path/to/mtp.gguf`
  for models without a welded nextn layer (e.g. bonsai PTQ1_0). With split
  CPU/GPU layers also pass `--kvmem-mtp-state snapshots` — the default
  `replay` mode requires all GDN layers on GPU.

## Troubleshooting

- `failed to create context ... GDN replay requires ...` — use
  `--kvmem-mtp-state snapshots`, or offload all recurrent layers to GPU.
- Server closes the connection mid-request with an assert in
  `ggml_backend_event_record` — fixed in kvmem-llama.cpp `5a18001`; update the
  adapter checkout.
