# proj_3：中文医疗领域适配 + 可复现推理服务

## 0. 先说结论

本项目的主线目标是：在 4×RTX 3090 上完成一个可复现的中文医疗领域适配与推理服务基线，回答两个工程问题：

1. CMB 领域微调是否能带来跨题库、非背题的有效提升？
2. 同一模型、同一请求集在不同量化格式和推理引擎下，延迟、吞吐和显存如何权衡？

主线只承诺：

- Qwen3.5-9B-Base + LoRA/QLoRA SFT；
- CMB answer-only 训练与严格隔离评测；
- 单引擎 vLLM 服务与可复现压测；
- 一个真实可加载的 W4A16/GPTQ/AWQ 量化格式对照；
- 通用能力保持与数据污染检查。

RLVR、SGLang、自研 INT8 和 mini engine 都是扩展项，不再作为 6～7 周主线的交付前提。

**简历主线建议**：

> 基于 Qwen3.5-9B-Base 构建中文医疗选择题领域适配管线，完成数据去重/选项重排/答案规范化、LoRA/QLoRA SFT、CMB 与跨题库评测，并在 4×RTX 3090 上完成 vLLM 量化服务和 TTFT/TPOT/吞吐/P99 压测。

Qwen3.5 官方模型卡：<https://huggingface.co/Qwen/Qwen3.5-9B>、<https://huggingface.co/Qwen/Qwen3.5-9B-Base>。

### 0.1 当前实现状态（2026-08-20 更新）

已实现并通过实测：CMB 下载/解压（含 zip-slip 防护）、schema 检查、split 去重（跨 split 撞题删训练侧）、选项重排与 answer-only SFT 构造、PEFT LoRA/QLoRA 训练、CMB 生成式/logprob 评测（候选池只由题型决定，无 gold 泄漏）、vLLM/SGLang 启动包装（默认只监听本机）、固定 trace 压测（TPOT 采用服务端 usage 真实 token）、训练守护（train/lora_watchdog.sh，独立 session，训练意外死亡自动 checkpoint 续训）。

4×RTX 3090（vip2023-2，conda env omni；依赖已 pin：torch 2.8.0+cu129 / transformers 5.15.0 / accelerate 1.14.0 / peft 0.20.0 / bitsandbytes 0.50.1，见 requirements-*.txt 与 pyproject.toml）实测记录：

- 全量 SFT 冒烟（train_full_smoke.py）：100/100 步 @4096 完成，7.15 s/步，loss 0.667→0.196，峰值 allocated 16.95GiB / reserved 21.3GiB；
- LoRA parity（lora_parity.py）：3/3 通过，logits 最大绝对差 0.0，可训练参数 43,278,336（0.48%）；
- LoRA 正式训练（2026-08-19 启动）：4×3090 FSDP2，73.5 s/步（grad_acc=16），1250 步/epoch，显存 13.8GiB/卡；
- 数据管线：CMB 去重 269,093→239,895（去 29,198），SFT 80,000 条 / 无效丢弃 0，val 280 条（manifest 见 out/manifests/）；
- 测试：21 个单测全部通过，ruff 0 错误。

尚未完成：vLLM 服务与压测真实数字、量化格式对照、跨题库/通用保持评测（M5-M6）。

### 0.2 快速开始

在 `proj_3` 目录执行。第一步只依赖 Python、`requests` 和 `tqdm`：

```bash
python -m unittest discover -s tests -v
python -m data.fetch_cmb
python -m data.inspect_cmb
python -m data.dedup_splits
python -m data.build_answer_sft
# 无 torch 环境必须显式 --char-fallback（字符近似口径）；有 tokenizer 的环境省略该参数，
# workload 会按真实 token 计量桶大小。
python -m serving.workload --output out/bench/workload.jsonl --per-bucket 10 --char-fallback
```

下载模型元数据：

```bash
python -m model.download --model-id Qwen/Qwen3.5-9B-Base --metadata-only
```

在目标 Linux 服务器下载完整权重：

```bash
python -m model.download --model-id Qwen/Qwen3.5-9B-Base
```

安装训练依赖后，4 卡主线使用 FSDP LoRA：

```bash
pip install -r requirements-train.txt
torchrun --standalone --nproc_per_node=4 -m train.train_lora \
  --model out/models/Qwen--Qwen3.5-9B-Base \
  --train-data data/processed/sft/train.jsonl \
  --eval-data data/processed/sft/val.jsonl \
  --output-dir out/checkpoints/lora \
  --max-length 4096
```

如果 4 卡显存或 Qwen3.5 量化内核不稳定，使用 QLoRA：

```bash
torchrun --standalone --nproc_per_node=4 -m train.train_lora \
  --model out/models/Qwen--Qwen3.5-9B-Base \
  --qlora --train-data data/processed/sft/train.jsonl \
  --eval-data data/processed/sft/val.jsonl \
  --output-dir out/checkpoints/qlora
```

训练 500～1000 step 后，先对 `val.jsonl` 跑 logprob early gate，再决定是否继续完整训练。

```bash
# 候选池只由 question_type 决定（多项选择题评全部 2..n 元组合），不使用 gold 宽度。
# 0-3 卡常被他人占用；实验统一跑 GPU 4-7。
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m eval.eval_cmb_logprob \
  --model out/models/Qwen--Qwen3.5-9B-Base \
  --data data/processed/cmb/val.jsonl \
  --output out/eval/base_val_logprob.json
```

训练异常中断时 train/lora_watchdog.sh 会自动用最新 checkpoint 续训（跑在独立 tmux session：
`tmux new-session -d -s lora_watch bash train/lora_watchdog.sh`）；手动续训用
`--resume-from-checkpoint out/checkpoints/lora/checkpoint-<N>`。

### 0.3 评审整改记录（2026-08-20）

一次外部评审指出 7 类问题，全部已修复并验证（未影响正在运行的训练）：

1. **logprob 评测答案泄漏**：候选池原按 gold 答案宽度生成（多选"选错个数永不判错"），现只由 `question_type` 决定——多选评全部 2..n 元组合（n=5 时 26 个候选），val 中 40/280 条多选题受影响；
2. **TPOT 用字符近似**：bench 现请求 `stream_options.include_usage`，TPOT/吞吐改用服务端回传的真实 completion_tokens；服务端不回传时降级字符口径并在结果留痕 `token_source`；workload 无 tokenizer 必须显式 `--char-fallback`，不允许静默降级；
3. **依赖未锁定**：requirements-train/-serve/-smoke.txt 与 pyproject.toml 已 pin 到 omni 实测版本（torch 2.8.0+cu129、transformers 5.15.0；vllm 0.27.1 待服务冒烟复核）；
4. **测试单薄**：新增 zip 解压安全、跨 split 去重、候选池、标签屏蔽、压测统计共 7 个测试类，21 个单测全通过；
5. **README 与实际状态漂移**：0.1 已按实测数字重写（本节即整改记录）；
6. **可移植性/安全**：serve 默认只监听 127.0.0.1（`--host` 显式指定才对外）；zip 解压拒绝绝对路径与 `../` 穿越成员；watchdog 的目录/conda/GPU 均支持环境变量覆盖；
7. **5 个 ruff 错误**：未使用导入已清理，ruff 0 错误。

---

## 一、基座模型与运行边界

### 1.1 主模型

| 项目 | 决策 |
|---|---|
| 主训练基座 | `Qwen/Qwen3.5-9B-Base` |
| 服务基线 | 原始 `Qwen/Qwen3.5-9B-Instruct` + 微调后的 Base/LoRA 模型 |
| 训练方式 | 16-bit LoRA 为稳妥主线；QLoRA 作为显存优先方案 |
| 模态范围 | 只做文本医疗问答，服务时启用 text-only 路径；不训练视觉模块 |
| 主要运行时 | Transformers + PEFT/bitsandbytes；vLLM 服务 |
| 可选运行时 | SGLang，仅在 vLLM 主线稳定后复测 |
| 备用基座 | 若 Qwen3.5 夜版依赖在目标机不稳定，切换 Qwen3-8B 或 Qwen2.5-7B；不改变数据和评测接口 |

### 1.2 为什么不把全量微调作为主线

9B BF16 权重约 18～20GB。权重、梯度和 8-bit Adam 状态的理想全局账约 54GB，FSDP 分片后理论上约 13.5GB/卡，但还要承担 all-gather、激活、临时张量、通信和 checkpoint 峰值。

因此：

- 全量 SFT 只保留为 100～300 step 的环境 smoke test（已在 4×3090 实测 100/100 步，见 0.1）；
- 正式实验使用 LoRA/QLoRA，优先保证可复现和可比较；
- 所有训练脚本记录 `max_memory_allocated`、`max_memory_reserved` 和有效 token 数；
- `max_seq_len` 主线先锁 4096，确认显存后再尝试 8192；不以 262K 上下文作为实验目标。

### 1.3 Qwen3.5 特有的实现约束

Qwen3.5-9B 是带 Vision Encoder 的混合架构，包含 Gated DeltaNet 与 full attention。自研 LoRA 不能假设所有可训练结构都是普通 Transformer attention 的 `q_proj/k_proj/v_proj/o_proj`。

主线必须先使用 PEFT/官方 Transformers 路径完成 parity test：

- 注入前后 trainable parameter 数量正确；
- LoRA merge/unmerge 后 logits 误差低于阈值；
- 保存/恢复后输出一致；
- text-only 输入不加载视觉模块或不为视觉模块错误分配显存。

自研 LoRA 只作为独立实验，不作为主线训练依赖。

---

## 二、数据方案：CMB answer-only 主线

CMB 官方数据卡：<https://huggingface.co/datasets/FreedomIntelligence/CMB>。

官方划分为：

- `CMB-Exam train`：269,359 条；
- `CMB-Exam val`：280 条，带 solutions/explanations，用于开发、few-shot 或 CoT 参考；
- `CMB-Exam test`：11,200 条，**不含答案**（官方 held-out 设计，答案不公开，无法本地评测 acc）；
- `CMB-Clin`：74 个复杂病例，作为定性案例，不承担强统计结论。

### 2.1 数据使用规则

| 数据 | 用途 | 规模/规则 |
|---|---|---|
| CMB-Exam train | 主 SFT 数据 | 从 269,359 条中抽取 80,000 条；不使用 val/test |
| CMB-Exam val | 开发集、格式调试、few-shot | 固定保留 280 条，不进入训练 |
| 训练未见 held-out | 最终主评测 | train 去重后 239,690 条中排除 SFT 用到的 80,000 条，余 159,690 条训练完全未见；seed 20260818 抽样 5,000 条（多选 493）→ data/processed/cmb/heldout_5000.jsonl（构建脚本 data/build_heldout.py） |
| 医疗对话 | 可选混合数据 | 最多占训练 token 的 10%；必须记录具体数据集、许可证和过滤比例 |
| 通用保持数据 | 防遗忘 | 项目一数学/通用数据最多占训练 token 的 5% |
| 教师解析 | 后续扩展 | 不进入主线；若使用，必须记录教师模型、生成参数、抽检结果和 API 成本 |

### 2.2 训练格式

主线采用 answer-only，不宣称 CMB train 自带解析：

```text
用户：题干 + 选项
助手：答案：B
```

解析输出作为后续实验，不作为主线验收条件。理由是 CMB train 的原始题目主要提供答案，强行补“答案 + 解析”会引入教师噪声或改变零 API 的成本假设。

### 2.3 必须实现的数据处理

1. 识别单项选择题和多项选择题；多选答案按排序后的集合规范化。
2. 选项重排时同步重映射答案标签，保存原始选项、重排种子和映射表。
3. 对 train/val/test 做精确去重和近重复检查；发现跨 split 重复时记录并移除训练侧样本。
4. 保留 `source_id`、原始哈希、处理版本、丢弃原因和许可证信息，生成 `manifest.json`。
5. 数据集构造结果必须提供：条数、token 长度分布、题型分布、学科分布、重排比例、去重比例和抽检样例。

“自建 10 万条数据”改称“自建数据构造与清洗管线”，避免把公开题目包装成原创数据。

---

## 三、评测方案

### 3.1 主评测：CMB-Exam

同时实现两种评测，避免只依赖生成文本解析：

1. **生成式 exact-match**：模型输出答案标签，处理空格、中文标点、大小写和多选集合。
2. **候选答案 logprob**：对每个候选答案分别计算条件 logprob，选择最高者；单选和多选分开实现。

报告内容：

- 训练未见 held-out acc（官方 test 无答案，见 2.1）；
- 单选/多选 acc；
- 6 个大类及 28 个子类 acc；
- 微调前、LoRA 后、量化后三者对照；
- bootstrap 95% CI；
- 选项重排一致性；
- 近重复样本剔除前后结果。

不预设“必须提升 10～20 点”，先跑基线再定义 effect size。

### 3.2 泛化与污染检查

- 固定一个明确版本的跨题库评测，例如 MedBench 或 CMMLU 医疗子集，不使用“MedBench 或其他题库”的模糊表述；
- CMB-Clin 只报告案例级定性结果和错误类型；
- 检查 CMB train 与公开 test 的近重复；
- 报告 CMB test 作为公开 benchmark 的局限，不能仅凭它证明获得了新医学知识。

### 3.3 通用能力保持

复用项目一的评测接口和答案抽取思想，不直接复用其 MiniMind 模型代码：

- GSM8K test 1,319 条；
- 项目一合成算术集作为补充；
- 微调前后使用完全相同的 prompt、采样参数和 token budget；
- 报告绝对变化、相对变化和置信区间；
- “掉点小于 2 点”是观察标准，不是硬性承诺。

---

## 四、代码结构（主线可运行，扩展项隔离）

```text
proj_3/
├── README.md
├── pyproject.toml              # 锁 Python / PyTorch / Transformers / PEFT / vLLM 版本
├── config.py                   # 模型、数据、长度、seed、路径（Paths/DEFAULT_*）
├── data/
│   ├── fetch_cmb.py            # 官方 HF 下载、缓存、校验和
│   ├── inspect_cmb.py          # 字段、题型、许可证和长度统计
│   ├── dedup_splits.py         # split 内/跨 split 精确去重与选项/答案规范化
│   ├── build_answer_sft.py     # answer-only chat 数据构造（选项重排 + 答案重映射）
│   ├── clean_dialogue.py       # 许可证过滤、角色校验、去重、长度过滤
│   └── build_manifest.py       # 数据版本和流失统计
├── model/
│   ├── download.py             # HF 权重下载（断点续传、sha256 校验、metadata-only）
│   ├── hf_loader.py            # AutoProcessor、chat template、text-only 加载路径
│   └── lora_parity.py          # PEFT 注入、零初始化/merge/保存恢复 logits parity test
├── train/
│   ├── train_lora.py           # 主线 LoRA/QLoRA SFT（--qlora 开关，FSDP、resume、显存日志）
│   ├── train_full_smoke.py     # 可选：FSDP 全量微调短冒烟（需 --confirm），不做正式交付
│   └── lora_watchdog.sh        # 训练守护：独立 session，训练死亡自动 checkpoint 续训
├── eval/
│   ├── eval_cmb_generate.py    # 生成式答案评测
│   ├── eval_cmb_logprob.py     # 候选答案 logprob 评测（候选池只由题型决定，无 gold 泄漏）
│   ├── eval_general.py         # GSM8K/合成算术保持（数值提取比较）
│   ├── eval_cross_domain.py    # 固定版本跨题库评测（复用 generate 的 run()）
│   ├── early_learning_gate.py  # logprob early gate 门控
│   └── report_metrics.py       # CI、分组统计（题型/大类）与 bootstrap 区间
├── serving/
│   ├── serve_vllm.py           # 主线 OpenAI 兼容服务（默认只监听 127.0.0.1）
│   ├── serve_sglang.py         # 可选双引擎复测（同样默认本机监听）
│   ├── bench.py                # 固定 trace 的 TTFT/TPOT/吞吐/P99/失败统计（TPOT 用服务端真实 token）
│   ├── quant_export.py         # 真实可加载量化格式（GPTQ）导出/校验
│   └── workload.py             # 短/中/长 token 桶与三档 prefix reuse 请求集
├── tests/
│   └── test_*.py               # 数据管线/zip 安全/候选池/标签屏蔽/压测统计（21 个单测）
├── optional/
│   ├── rlvr/                   # 条件满足后再实现 GRPO/verl 对照（方差 smoke 已就位）
│   ├── custom_int8/            # parity、误差和离线性能实验
│   └── toy_engine/             # 尚未实现：不与 Qwen3.5 同模型做主结论（见 6.2）
├── scripts/
│   ├── run_baseline.sh
│   ├── run_lora.sh
│   ├── run_eval.sh
│   ├── run_serve.sh
│   └── run_bench.sh
└── out/
    ├── manifests/
    ├── checkpoints/
    ├── eval/
    └── bench/
```

说明：`configs/*.yaml` 已由 `config.py` 替代；`train_qlora.py` 已并入 `train_lora.py --qlora`；
自研 LoRA（`custom_lora_lab.py`）与 toy engine 均未实现，主线使用 PEFT parity 与 vLLM/SGLang。

---

## 五、部署与压测设计

### 5.1 服务配置

Qwen3.5 vLLM 支持说明：<https://docs.vllm.ai/en/stable/models/supported_models/>。

主线服务必须固定：

- 模型 commit hash；
- Transformers、PyTorch、CUDA、vLLM commit/nightly 版本；
- `--language-model-only` 或等价 text-only 设置；
- `max-model-len=4096` 起步，确认显存后再扩展；
- 相同 tokenizer、chat template、采样参数和停止条件。

### 5.2 量化对照

首选一个能被 vLLM 真实加载的官方或明确来源 checkpoint，例如 W4A16/GPTQ/AWQ 之一；不要把“自己实现 per-channel round-to-nearest”直接称为生产 INT8。

量化主线只比较：

1. BF16 text-only；
2. 一个固定的 W4A16/GPTQ/AWQ checkpoint；
3. 量化前后 CMB acc、logprob acc、显存和吞吐。

自研 INT8 后续必须包含：校准集、权重/激活 scale、逐层误差、端到端 logits parity，以及真实推理 kernel 或可加载导出格式；否则只作为离线算法实验。

### 5.3 压测请求集

第一版不用模糊的“agentic 负载”，而是先生成固定 trace：

- 并发：1/4/8/16；
- 输入：短 256、中 1024、长 3072 token；
- 输出：32/128/512 token；
- prefix：无复用、固定 system/tool schema 复用、长前缀高复用；
- 采样：greedy 和固定 seed 的 sampling 分开；
- 每组：预热 20 次，正式 100 次，记录失败/超时；
- 指标：TTFT、TPOT（服务端 usage 回传的真实 completion_tokens，`stream_options.include_usage`；服务端不回传时降级为字符口径并在结果中留痕 `token_source`）、端到端延迟、吞吐、p50/p95/p99、GPU 显存和功耗。

只有固定 trace 稳定后，才加入多轮工具调用，并同时报告请求级 prefix reuse 比例和引擎级 cache 指标。

---

## 六、RLVR 与自研组件：明确降级为扩展项

### 6.1 RLVR 条件

只有满足以下条件才启动 RLVR：

- SFT 后 CMB test、跨题库和通用保持结果稳定；
- 4×3090 上完成一次 LoRA policy 的短 rollout smoke；
- policy、冻结 reference、rollout、optimizer 的显存分配有实测记录；
- `old_logp`、`new_logp`、`ref_logp`、KL、reward、response length 全部 finite；
- 组内 reward 不是长期全对或全错。

第一版 RLVR 使用短输出、小 group、LoRA policy 和冻结 reference。不能复用“当前 policy 同时作为 reference”的实现作为最终 KL 约束。

验收指标不再写死为“Y%→Z% 必须提升 3 点”，而是报告：是否提升、提升是否跨 seed 稳定、是否只提升 CMB 训练分布、是否损害通用能力。

### 6.2 mini engine

不再承诺用约 500 行 mini engine 正确实现 Qwen3.5 并与 vLLM/SGLang 做同模型结论。可选方向只有两个：

- 换用纯 Transformer 基座，单独研究 continuous batching/paged KV；
- 保留 Qwen3.5，但把 toy engine 明确标为教学演示，不写入生产性能王牌图。

---

## 七、实施阶段与时间线

| 周期 | 主线内容 | 交付物 | 通过条件 |
|---|---|---|---|
| 第 1 周 | 环境锁定、模型下载、text-only 单卡加载、CMB schema 检查、原始 baseline | `pyproject.toml`、baseline 日志、版本表 | 32 条样本生成与答案解析正确 |
| 第 2 周 | CMB 去重、选项重排、answer-only 构造、manifest、抽检 | 80k 训练集、280 条 dev、统计报告 | split 无污染；单/多选标签全通过单测 |
| 第 3 周 | LoRA SFT、单卡 smoke、4 卡短 smoke、checkpoint resume | 可恢复 checkpoint、显存峰值日志 | loss finite；保存/恢复 logits 一致 |
| 第 4 周 | CMB generate/logprob、通用保持、跨题库评测 | 评测 JSON、CI、分组结果 | 全量 test 可复现，结果不依赖随机输出格式 |
| 第 5 周 | vLLM text-only 服务与固定 trace 压测 | 服务脚本、bench JSON、p50/p95/p99 | 并发 1/4/8/16 无系统性失败 |
| 第 6 周 | 一个量化格式、量化精度与吞吐对照 | 量化 checkpoint、对照报告 | 量化模型可真实加载；精度损失可解释 |
| 第 7 周 | 可选 SGLang 复测或 RLVR smoke | 扩展实验报告 | 不影响主线结果；失败也记录归因 |

### 主线验收门槛

- 数据 manifest、处理脚本和随机种子齐全；
- 单卡 smoke 与 4 卡 smoke 都能运行；
- LoRA merge/unmerge 和 checkpoint resume 通过 parity test；
- CMB 评测同时支持生成式和 logprob；
- 服务端与客户端固定版本、token budget、warmup 和请求 trace；
- 所有结果都保存原始 JSON/日志，而不是只保留图片或手填数字。

---

## 八、风险与合规

| 风险 | 等级 | 对策 |
|---|---|---|
| Qwen3.5 依赖夜版不稳定 | 高 | 锁 commit；保留 Qwen3-8B/Qwen2.5-7B fallback |
| CMB 背题或跨 split 近重复 | 高 | 去重、选项重排、跨题库、报告公开 benchmark 局限 |
| 量化 checkpoint 与 vLLM 不兼容 | 高 | 先加载验证，再开始 benchmark；不把离线自研格式冒充生产格式 |
| LoRA 漏掉 Qwen3.5 混合结构 | 高 | PEFT 主线、参数审计、merge/unmerge logits parity |
| 医疗对话数据许可证不清 | 中 | 只保留明确许可证来源，README 写明 dataset/version/license |
| 医疗回答造成误用 | 中 | demo 显著免责声明；只做教学与模型评测，不提供诊疗建议 |
| 4×3090 实测环境差异 | 中 | 固定驱动、CUDA、NCCL、温度和功耗；报告硬件信息和显存峰值 |

---

## 九、最终项目定位

项目完成主线后，可以稳定证明：

1. 能够为公开医疗题库构造可审计的数据处理和训练管线；
2. 能够在有限显存上完成 9B 级模型的参数高效领域适配；
3. 能够用严格 split、logprob、跨题库和通用保持评估模型，而不是只报一个漂亮 acc；
4. 能够把同一个模型导出到真实推理服务，并用固定 workload 给出延迟/吞吐/显存权衡。

这已经足够成为一个完整的算法工程项目。RLVR、自研 INT8 和推理引擎应作为有实验结果再写入简历的加分项，而不是项目能否成立的前提。

---

## 十、里程碑

- [ ] M1：环境与依赖已锁定、text-only 4 卡加载 ✅（parity/全量冒烟实测）；待补 32 条生成 smoke 与原始 CMB baseline 记录。
- [x] M2：CMB 80k answer-only 数据、跨 split 去重、选项重排、多选规范化、manifest 全部完成（manifest 见 out/manifests/，单测通过）。
- [x] M3：LoRA 训练完成（2026-08-19 启动 → 08-21 完成，checkpoint-1250，eval_loss 0.1139@500 → 0.1052@1000 → 0.10396@1250）；parity 3/3 ✅；watchdog 自动续训已实测（checkpoint-500 恢复、loss/lr 连续性验证）；FSDP2 adapter 加载修复（强制 text_only，248/248 lora_B 非零）；merge parity ✅（fp32 下 merge 前后 max|Δlogits| = 1.9e-5 < 1e-4，bf16 下 argmax 20/20 一致、~0.2 的差异为 bf16 权重折叠舍入，符合 W/128×√4096 量级）。
- [ ] M4：进行中——val（280 条）LoRA 72.50% vs base 70.71%（+1.79，不显著）；**heldout 5,000 条（训练未见，seed 20260818）LoRA 81.68% vs base 76.94%，Δ=+4.74pt，bootstrap 95% CI [+3.82,+5.66]，McNemar p=7.2e-25**；单选 +3.42pt（p=3.7e-14）、多选 +16.84pt（p=2.1e-14，493 条）；待补 6 大类/28 子类 breakdown 与选项重排一致性。
- [ ] M5：未开始（serve_vllm/bench 入口就绪，TPOT 已改为服务端真实 token 计量）。
- [ ] M6：未开始。
- [ ] M7：未开始（扩展项，不影响主线交付）。
