// Adapt the kvmem vision tool to the newer (prismml) mtmd API without editing the
// shared kvmem repo. Drift points in kvmem-vision.cpp:
//   - mtmd_helper_init_opt_default() no longer exists; process_mtmd_prompt takes
//     bool is_placeholder in place of the init_opt argument
//   - mtmd_helper_decode_image_chunk_with_decoder is declared in mtmd-helper.h,
//     which server-common.h no longer pulls in
#include "mtmd-helper.h"
#define mtmd_helper_init_opt_default() false
#include "kvmem-vision.cpp"
