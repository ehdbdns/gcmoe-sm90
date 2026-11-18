# 单层 decode trace（EPLB 重排之后）— 可直接拖进 Perfetto

文件：**`one_decode_layer_postrearr.perfetto.json`**（11.7 KB）

打开方式：<https://ui.perfetto.dev> → 左侧 **"Open trace file"** 选中该文件。

## 为什么这一层是"重排之后"

torch profile 里共 **4600 次** gc_moe 调用。EPLB 重排发生在第 4359 次之后
（那里有 **159.4 ms** 的停顿，缓冲区正在重建），因此：

- `call < 4360` = 重排前
- `call ≥ 4360` = **重排后**（共 240 次：post-rearrange warm + profile 窗口）

本文件取 **call 4360**——重排后第一个完整层，且 8 个 rank 的 dispatch 起点
只差 **124 µs**（重排后最紧凑的一次）。

## 轨道布局：和 `phases5_role4_trace.png` 完全一致

每个 (rank, role) 组合是**独立的一条轨道**，共 40 条，用
`process_sort_index` / `thread_sort_index` 强制成图中顺序
（rank0 dispatch → rank0 shared → … → rank7 combine），
每根 bar 的**名称里直接写着阶段描述和耗时**，所以 Perfetto 会把文字画在条内。

| tid | 角色 | 启动几何 |
|---:|---|---|

| 顺序 | 角色 | 启动几何 |
|---:|---|---|
| 1 | role1 dispatch（routing/count-exchange + token pull） | grid=24 block=256 |
| 2 | **role4 shared expert L1+L2** | grid=71 block=256 |
| 3 | role5 ECO plan | grid=1 block=512 |
| 4 | role2 routed GEMM | grid=71 block=384 |
| 5 | role3 combine | grid=288 block=384 |

每一根 bar 都是 profile 里**真实的 gc_moe kernel 时间区间**，没有任何合成。
`wait_for(role4)` 的箭头（role4 结束 → role2 开始）标出 `wait_for=(4,)` 依赖。

## 这一层的数值（µs）

| rank | dispatch | shared | plan | routed | combine | role2起 − role4止 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1332 | 694 | 149 | 3026 | 531 | **1.0** |
| 1 | 522 | 700 | 144 | 2620 | 907 | **0.9** |
| 2 | 1505 | 698 | 128 | 3462 | 64 | **0.9** |
| 3 | 1263 | 700 | 125 | 3015 | 512 | **0.9** |
| 4 | 722 | 692 | 229 | 2669 | 982 | **1.1** |
| 5 | 708 | 690 | 228 | 2680 | 969 | **1.0** |
| 6 | 650 | 702 | 240 | 2633 | 1019 | **1.2** |
| 7 | 740 | 700 | 231 | 2892 | 752 | **0.9** |

## 三个可在 Perfetto 里直接看出来的结论

1. **role4 是 graph 根节点，与 dispatch 并行**：role4 在 dispatch 开始后
   11.3–15.8 µs 就启动，整段落在 dispatch 窗口内（占其 46–97%）。
2. **role2 严格被 role4 门控**：`role2 起点 − role4 结束` 在 8 个 rank 上都是
   **0.9–1.2 µs**，8 个 rank 无一例外。
3. **shared 计算量各 rank 相等**：690–702 µs，因为 role4 不做 TP 切分，
   每个 rank 算一份完整 shared（对比 vLLM marlin 路径的 216 µs，那是 TP/8 分片）。

## 注意

本文件的 5 条轨道只含 gc_moe 的 kernel。shared expert 已由 role4 接管，
所以看不到 vLLM 的 marlin/nvjet；如果看的是重排前或 role4 未启用的运行，
shared 会出现在 aux stream（stream 51）上的 marlin 里。

## 图片与 Perfetto 文件的关系

`phases5_role4_from_perfetto.png` **直接读 `one_decode_layer_postrearr.perfetto.json` 画出来的**，
脚本是同目录的 `plot_from_perfetto.py`：

```
python plot_from_perfetto.py one_decode_layer_postrearr.perfetto.json out.png
```

所以图里的每一根条都对应 JSON 里的一个 `ph:"X"` slice，数值必然一致。

关于 Perfetto 界面：它只按 slice 的 `name` 渲染文字，且轨道背景/条高/字体是它前端固定的，
**无法做到和这张图完全一样的观感**。要"像图一样"请看 PNG；要在时间轴上交互排查请用 Perfetto。
