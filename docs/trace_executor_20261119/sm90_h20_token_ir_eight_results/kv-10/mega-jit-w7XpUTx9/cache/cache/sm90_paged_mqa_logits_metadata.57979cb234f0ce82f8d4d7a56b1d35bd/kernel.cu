
#include <deep_gemm/scheduler/sm90_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sched::sm90_paged_mqa_logits_metadata<
        96, 256, 78, false
    >);
};
