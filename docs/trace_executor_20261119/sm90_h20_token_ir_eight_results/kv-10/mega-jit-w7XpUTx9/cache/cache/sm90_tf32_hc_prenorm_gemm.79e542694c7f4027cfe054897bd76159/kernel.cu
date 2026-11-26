
#include <deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm90_tf32_hc_prenorm_gemm_impl<
        24, 5120,
        64, 32, 64,
        16,
        128,
        12,
        128, 128
    >);
};
