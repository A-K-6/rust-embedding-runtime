#ifndef RUNTIME_AOCL_EVAL_CHECKED_SHIM_H
#define RUNTIME_AOCL_EVAL_CHECKED_SHIM_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct runtime_aocl_packed runtime_aocl_packed;
/* Return 0=success, positive=AOCL dlp_clsc_err_t, negative=shim error:
 * -1 invalid pointer/shape/length/threads, -2 allocation, -3 invalid size,
 * -4 missing AVX2/FMA, -5 thread-count setup mismatch.
 * Arrays must remain valid for duration of call; output must not alias inputs.
 * Only create copies/reorders the weight. Free source W after successful create.
 * The opaque handle is immutable until destroy; destroy requires exclusive use.
 */
int runtime_aocl_packed_create(const float *w, size_t w_len,
    int64_t n, int64_t k, int threads, runtime_aocl_packed **out,
    uint64_t *packed_bytes);
int runtime_aocl_packed_forward(const runtime_aocl_packed *h,
    const float *a, size_t a_len, int64_t m, float *c, size_t c_len);
void runtime_aocl_packed_destroy(runtime_aocl_packed *h);
int runtime_aocl_available(void);
#ifdef __cplusplus
}
#endif
#endif
