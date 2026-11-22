"""Add device-clock phase envelopes to the two private, already clocked backends."""
from pathlib import Path
import sys


def replace(text, anchor, value):
    if text.count(anchor) != 1:
        raise ValueError(f"Unexpected upstream source at {anchor!r}")
    return text.replace(anchor, value)


def mega(text):
    def before(anchor, code):
        nonlocal text
        text = replace(text, anchor, code + "\n" + anchor)

    text = replace(text, "if (field < 2)", "if (field < 2 || (field >= 4 && field % 2 == 0))")
    before("        // Adjust registers\n        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();",
           "        if (thread_idx == 0) matrix_stamp(4);")
    edge = "/* After the NVLink barrier, there is a grid sync */ true\n        );"
    text = replace(text, edge, edge + "\n        if (thread_idx == 0) matrix_stamp(5);")
    before("        uint32_t pull_mbarrier_phase", "        if (lane_idx == 0) matrix_stamp(6);")
    before("        // Clean workspace for the next usage", "        if (lane_idx == 0) matrix_stamp(7);")
    first = "                    if (k_block_idx == 0 && !task_info.is_shared()"
    before(first, """                    if (k_block_idx == 0 && cute::elect_one_sync()) {
                        auto phase = task_info.block_phase;
                        matrix_stamp(phase == sched::BlockPhase::SharedLinear1 ? 8 :
                                     phase == sched::BlockPhase::SharedLinear2 ? 10 :
                                     phase == sched::BlockPhase::Linear1 ? 12 : 14);
                    }""")
    before("                    if (task_info.is_shared()) {\n                        ptx::red_add_rel",
           "                    matrix_stamp(task_info.is_shared() ? 9 : 13);")
    before("                // L2 BF16 epilogue: write GEMM output",
           "                if (!task_info.is_shared() && epilogue_warp_idx == 0 && cute::elect_one_sync()) matrix_stamp(16);")
    before("                if (!task_info.is_shared() && epilogue_warp_idx == 0 && cute::elect_one_sync())\n                    matrix_stamp(2);",
           "                if (epilogue_warp_idx == 0 && cute::elect_one_sync()) matrix_stamp(task_info.is_shared() ? 11 : 15);")
    before("        // Combine: reduce top-k", "        if (lane_idx == 0) matrix_stamp(18);")
    return text


def gc(root):
    layout = root / "gc_moe/gc/__init__.py"
    layout.write_text(replace(layout.read_text(), "size = self.clock_offset + 32", "size = self.clock_offset + 19 * 8"))
    path = root / "gc_moe/impls/sm100_fp8_fp4_gc_moe.py"
    text = path.read_text()

    def mark(anchor, field, condition, indent, end=False):
        nonlocal text
        op = "max" if end else "min"
        pad = " " * indent
        code = (f"{pad}if cutlass.const_expr(self.clock_offset is not None):\n"
                f"{pad}    if {condition}:\n"
                f"{pad}        red_global_{op}_u64(signals.iterator + self.clock_offset // 4 + "
                f"({field}) * 2, read_globaltimer())\n")
        text = replace(text, anchor, code + anchor)

    mark("            # First count local routes", 4, "tid == 0", 12)
    edge = "            # GCMoEScheduler::fetch_expert_recv_count"
    mark(edge, 5, "tid == 0", 12, True)
    mark(edge, 6, "lane == 0", 12)
    mark("            if cutlass.const_expr(self.eco_trace):\n                if elect_one_sync():\n                    self._eco_trace(signals, 8, 2)",
         7, "lane == 0", 12, True)
    first = "                            # Keep descriptor layout bits fixed"
    last = "                    is_w1_release = task_phase"
    for phase, field in (("SharedLinear1", 8), ("SharedLinear2", 10), ("Linear1", 12), ("Linear2", 14)):
        match = f"task_phase == Uint32(GCBlockPhase.{phase})"
        mark(first, field, f"k_block == 0 and {match} and elect_one_sync()", 28)
        mark(last, field + 1, f"{match} and physical_warp == 4 and elect_one_sync()", 20, True)
    mark("                            # Source-contract W2 vector epilogue", 16,
         "task_phase == Uint32(GCBlockPhase.Linear2) and physical_warp == 4 and elect_one_sync()", 28)
    mark("                self._role3_combine(signals", 18, "lane == 0", 16)
    path.write_text(text)


if __name__ == "__main__":
    gc(Path(sys.argv[1]))
