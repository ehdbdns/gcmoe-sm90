
#include <deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_paged_mqa_logits<
        1, 32,
        128, 32,
        true, false,
        3, 3,
        256,
        128, 512,
        1,
        float
    >);
};
