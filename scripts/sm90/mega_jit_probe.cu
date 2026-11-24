// First arm-1 capture geometry from the H20 receipt, DeepGEMM 7292e7b.
#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>

void instantiate_mega_phase() {
    using namespace deep_gemm;
#if MEGA_PHASE == 1
    auto ptr = &sm90_fp8_mega_moe_l1_impl<
        2048, 5120, 2304, 384, 6, 48, 64, 128, 128, 107520, 215040,
        215040, 7, 64, 64, 256, 78, 8, 10.0f, true, true, false, false>;
#elif MEGA_PHASE == 2
    auto ptr = &sm90_fp8_mega_moe_l2_impl<
        2048, 5120, 2304, 384, 6, 48, 64, 128, 128, 107520, 215040,
        215040, 7, 64, 64, 256, 78, 8, 10.0f, true, true, false, false, false, false>;
#else
#error Select Mega phase 1 or 2
#endif
    (void)ptr;
}
