# [SM100] BS512 共同 KV 公平边界与同输入专家重放回执

执行者：SM100。未执行或混写 SM90 任务。  
任务书：`docs/sm100_megamoe/V41_BS512_FAIR_BOUNDARY.md`  
任务提交：`f91491499e122ac51b3c27eb799667f4a92e6e27`  
任务书 SHA256：`a0d59361aa4d85e8e462eb01f6ba53fbf6b8241f3335a33b639dd0962dc4c8d2`  
执行 revision：`f91491499e122ac51b3c27eb799667f4a92e6e27`，另加保持语义的插桩修复 `6196c0253258af115f2e68e3875d720537727897`（diff 在结果目录）  
GC 上游：`fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`  
结果目录：`docs/sm100_megamoe/v41_fair_boundary_results/`

## 结果

两臂短跑 exit code 均为 `0`。每臂执行 3 个 BS512 原生 greedy decode step；每 rank 实际记录 `120 = 3 x 40` 次整图 MoE 调用，block M 全部为 16。Mega 无 SM split；GC 每 rank 均为 dispatch 20 / compute 128。CPU 判读退出码为 `3`，`short_strict_pass=false`；按任务书没有启动正式 1000+1000，也没有性能结论。

共同 prefill 和共同 KV 控制成立：两臂 `prefill.json`、起点 token 和请求顺序一致，prompt SHA256 均为 `b4fee0121ad43c17fdd5fccf5c4809602468725776257facc16f867e4c25cd86`。8 个 rank 共用同一份 32 GiB checkpoint；GC source/readback SHA256 与 Mega 完全一致，controls 也一致。旧锚点 **step 0 / L3 / rank 7** 的采样输入、IDs、gate 与输出已全部一致，原先该锚点的分叉消失。

新的最早自然轨迹差异是 **step 0 / L7 / rank 6 / 专家 BF16 输出 `y`**：`y[24,4991]` 为 Mega `-6.705522537231445e-07`、GC `-6.742775440216064e-07`。该点在 attention 摘要、专家 hidden、top-k IDs 和 gate 相同的条件下出现，因此不是输入量化或路由差异。

step 0 共 5 个差异，全部仅 `y`：L7/rank6、L16/rank3、L33/rank0、L37/rank4、L39/rank5。step 1 差异扩展到 attention，最早 `attn_in` 为 L35/全部 rank，最早 `hidden_states` 与 `topk_weights` 为 L34/rank4；`topk_ids` 直到 step 2 / L36 / rank7 才首次不同。全部记录为 82 个原生差异点（step 0/1/2 = 5/42/35），覆盖 8 个 rank。

输出 token 仅 1/1536 个不同：request `281`、step `2`，Mega `22`、GC `23`。全局 step/layer 槽负载仍为 3072，但 rank 1/4/5 在 step 2 分别有 1/1/2 个槽计数差。

## 同输入重放

每 rank 120 个重放点，共 960 个。15 个点被标记，step 0/1/2 各 5 个。15 个点的量化输入、hidden、top-k IDs、gate、`x`、`sf`、`shared_sf` 全部相同，7 个 routed/shared slot 的 partial hash 也全部相同；Mega 自回放与自然图 `y` 逐元素相同。相同输入和相同 slot partial 下，GC 的最终 `y` 仍有 1--9 个 BF16 元素不同；首例仍是 step 0 / L7 / rank 6，最大绝对差 `3.725290298461914e-09`。该结果把剩余差异收窄到 slot partial 之后的最终 BF16 归约/写回路径，而不是专家输入或路由。

同时发现 15 个标记点的 Mega `partials_verified=false`、GC 为 `true`。即 Mega 的 slot 求和未能由本诊断公式逐位复原实际输出；这是本次 strict 失败的另一项证据/限制，不是 partial hash 不同。

## 判据

37 项检查中 33 项通过。失败项为 `tokens`、`rank1_counts`、`rank4_counts`、`rank5_counts`；另有 82 个原生差异和 15 个同输入重放差异，所以不满足 `short_strict_pass`。逐项结果见 `checks.tsv`，完整判读见 `results/fair_boundary.json`。

## 证据与释放

原始 `native-*.pt`、`replay-*.json`、stats 数组和共同 KV 保留在 `/data0/home/admin/workspace/megamoe/v41-fair-boundary-nmRZZ9Q9/run/bs512`。`raw-manifest.tsv` 登记 130 个文件、277,414,112,145 bytes 的绝对路径、字节数和 SHA256；KV 使用 checkpoint manifest 中经双端读回验证的 SHA256。清单 SHA256 为 `3fec6e013cc97969dd61c3a7f54e1717ad536d5ecd897c141cca880d748c9fa8`。

本轮唯一代码修复是把插桩层的枚举从 V4 改为实际 V4.1 decoder layer，并按 `layer_id` 排序；无语义变更，`codefix.diff` SHA256 为 `b6551e0d45ba4cf4934220c1af271fceafa3665307f2d27193d0e18aed549b89`。上游最终 diff 在 `upstream.diff`。

八卡组锁于 `2026-11-23T20:12:07+08:00` 获取、`2026-11-23T20:37:46+08:00` 释放。释放后无 compute app、无 `/dev/nvidia*` 持有者，组锁可立即获取，见 `gpu_release_after_run.txt`。短程插桩不计入性能。
