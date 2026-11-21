# SM100 回执：V4.1-Flash 整模原生推理入口与安装来源

执行者：SM100。全程 CPU-only，`CUDA_VISIBLE_DEVICES=''`；未启动模型、未安装依赖、未修改共享环境或其他人的文件。外部源码只记录路径和行号，不提交源码正文。

## 结论

- 当前 RTP checkout `c207b52b2ddce5acf964514968717a748053973b` **未注册** `DeepseekV41ForCausalLM`；按任务书原命令检索时还因根目录没有 `rtp_llm/` 返回两条 `No such file or directory`，原始 stderr 已保留。
- 当前 checkout 的最近入口是 `DeepseekV4ForCausalLM`：`githubopensource/rtp_llm/model_factory_register.py:204`、`githubopensource/rtp_llm/models/deepseek_v4.py:1173`。
- admin 可读的历史 checkout `/data0/data3/wjh528431/work/dsv41-mega-gate-20261012/internal/github-opensource` **已注册精确 V4.1 入口**：`rtp_llm/model_factory_register.py:207`、`rtp_llm/models/deepseek_v41.py:261`。该轮记录的 revision 为 `a5f09e3c4fd3de584f88541586e89780e11c5844`。
- 模型权重路径 `/mnt/hf3fs/3fs/models/DeepSeek-V4.1-Flash` 存在；其 `config.json` 为 `model_type=deepseek_v41`、`architectures=["DeepseekV41ForCausalLM"]`。
- 本地 vLLM 源码 `/data0/data3/wjh528431/work/vllm`（commit `36f397281295f5b58852a7eaf429b90348161079`，metadata 版本 `0.1.dev1+g36f397281.precompiled`）识别 V4.1：`vllm/model_executor/models/registry.py:371`；但 `.venv/bin/python` 指向当前 namespace 不存在的 `/home/wjh528431/...`，不能直接复用。
- 当前 admin Python 环境：Python 3.12.7、torch 2.14.1+cu130；`rtp-llm`、`vllm`、`sglang`、`flashinfer-python`、`transformers` 均 missing。
- 公开 PyPI 的 `vllm==0.31.0` 有 x86_64、aarch64 wheel 和 sdist，文件与 SHA256 已记录；未下载、未安装。
- `rtpllm-migration:dsv4_20261012_200340` 容器有 Python 3.10.9、torch 2.11.0，但未安装 RTP-LLM；只读挂载历史 V4.1 源码后，`find_spec` 可解析 `rtp_llm.models.deepseek_v41` 与 `rtp_llm.models_py.model_desc.deepseek_v41_model`。最新 SM10x 镜像 `2026_11_19` 有 `rtp-llm 0.2.0`，但没有精确 V4.1 注册。

## 可复用入口

- V4.1 历史启动脚本：`internal/prefill-41.sh`、`internal/decode-41.sh`，使用 `--model_type deepseek_v41`、`--checkpoint_path`/`--tokenizer_path`、`--load_method fastsafetensors`、`--tp_size 4 --ep_size 4 --world_size 4`。
- 无凭证启动模板、镜像路径、源码路径和版本证据见 `v41_runtime_results/runtime_assets.md`。该模板未执行，不能据此宣称整模推理已通过。
- MegaMoE 当前入口：`mega_moe.py:166` 的 `MegaMoeExecutor`、`mega_moe_se.py:54` 的 `MegaMoeSEExecutor`、`strategy/fp8_fp4.py:55` 的 `mega_moe_se` 策略；完整前 65 条路径见 `mega_moe_matches.txt`。

## 产物

`docs/sm100_megamoe/v41_runtime_results/` 下的 `README.md`、`rtp.txt`、`rtp_command.txt`、`mega_moe_matches.txt`、`python.txt`、`vllm-wheel.txt`、`vllm_local.txt`、`container_inspect.txt`、`runtime_assets.md` 与 `SHA256SUMS`。

本回执仅证明入口、版本和安装来源；未做权重加载或端到端启动。
