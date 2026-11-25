
#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_mega_moe_l2_impl<
        512,
        5120, 2304,
        448, 6,
        7,
        64, 128, 128,
        35328,
        70656, 70656,
        7,
        64, 64, 256,
        78, 8,
        0x1.4p+3f,
        true,
        true,
        false,
        false, false, false
    >);
};
