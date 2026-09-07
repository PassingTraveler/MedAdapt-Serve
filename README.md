# MedAdapt-Serve

Qwen3.5-9B 中文医疗领域适配与可审计推理部署工程。

MedAdapt-Serve 在 4×RTX 3090 上完成从 CMB 数据治理、LoRA SFT、跨题库评测到 GPTQ 量化与 vLLM 服务压测的完整主线，回答两个工程问题：CMB 领域微调能否带来跨题库、非背题的有效提升；同一模型在不同量化格式和服务配置下延迟、吞吐和显存如何权衡。

> 这是教学与工程研究归档，不提供个体诊疗建议。官方 CMB test 不含答案，主评测使用训练完全未见的自建 held-out 与跨题库 CMExam；CMB 上的提升不能证明获得了新医学知识。

## Results

在同一模型、同一 prompt 与解码配置下：

- held-out 5,000 条（训练未见，seed 20260818）：LoRA merged 81.68%，Base 76.94%，+4.74pt，bootstrap 95% CI [+3.82, +5.66]，McNemar p=7.2e-25
- 单选 +3.42pt、多选 +16.84pt；6 个大类全部提升，12 个子类无一下降
- 跨题库 CMExam 6,145 条（与 CMB train 零重叠校验）：81.16% vs 78.54%，+2.62pt，CI [+1.81, +3.43]，McNemar p=2.9e-10
- GSM8K test 1,319 条：85.52% vs 86.43%，−0.91pt，CI [−2.58, +0.76] 含零，无显著通用能力损失
- GPTQ W4A16（group=128）：val 280 条 acc 72.86% vs BF16 71.07%，差值在量化噪声量级；权重 7.28 GiB vs 21.7 GiB
- 单卡单并发生成吞吐 110.8 vs 45.8 token/s（2.4×），TPOT p50 8.4 vs 21.1 ms
- 13 档固定 trace 压测（greedy + seeded，并发 1/4/8/16）全部 0 失败

完整数字、统计口径和已知局限见 [docs/RESULTS.md](docs/RESULTS.md)。

## Demo

- `python demo/app.py`：零依赖本地页面，回放 held-out、CMExam、GSM8K 评测汇总与 13 档压测的真实 JSON；不加载模型、不伪造在线生成。
- 接入真实 vLLM 服务后可做实时结构化选择题推理：

```bash
python -m serving.serve_vllm --model <gptq_or_bf16_path> --host 127.0.0.1 --port 8000 --served-model-name medadapt-gptq
python demo/app.py --api-base http://127.0.0.1:8000/v1 --model medadapt-gptq
```

演示边界见 [demo/README.md](demo/README.md)。

## What is implemented

### Data pipeline

- CMB 下载与安全解压（zip-slip 防护）、schema 检查
- split 内与跨 split 精确去重（269,359 → 239,690，跨 split 撞题删训练侧）
- 选项重排与答案重映射、多选集合规范化、answer-only SFT 构造（80,000 条）
- 数据 manifest：条数、token 长度分布、题型/学科分布、重排与去重比例、丢弃原因

### Training

- PEFT LoRA/QLoRA SFT，4×3090 FSDP2，可训练参数 43,278,336（0.48%）
- LoRA 注入/merge/保存恢复的 logits parity（merge 前后 max|Δlogits| 1.9e-5，fp32）
- checkpoint resume 与 watchdog 自动续训（训练意外死亡时用最新 checkpoint 恢复）
- 21 个单元测试：zip 安全、跨 split 去重、候选池、标签屏蔽、压测统计

### Evaluation

- 生成式 exact-match 与候选答案 logprob 双协议，避免只依赖文本解析
- 候选池只由题型决定（多选评全部 2..n 元组合），不使用 gold 宽度，无答案泄漏
- 跨题库（CMExam）与通用保持（GSM8K）对照；CMExam 使用 vLLM structured outputs 约束输出格式
- bootstrap 95% CI、McNemar 检验、题型/大类分组统计、选项重排一致性

### Serving and quantization

- vLLM OpenAI 兼容服务，默认只监听 127.0.0.1
- GPTQ W4A16 量化导出与校验（含 Qwen3.5 混合架构的导出适配与内核选择）
- 固定 trace 压测：3 输入长度 × 3 prefix 模式 × 并发 1/4/8/16，TTFT/TPOT/吞吐/p99
- TPOT 采用服务端 usage 回传的真实 completion_tokens；服务端不回传时降级字符口径并在结果中留痕
- GPU 显存与功耗全程监控

## Main experiment chain

当前最终结果对应的主线为：

```text
Qwen3.5-9B-Base
    ↓ CMB 去重、选项重排、answer-only 构造（80,000 条）
LoRA SFT（4×3090 FSDP2，1250 步/epoch）→ merge
    ↓
held-out 5,000 / CMExam 6,145 / GSM8K 1,319 评测
    ↓
GPTQ W4A16 导出（校准 CMB train 512 条）
    ↓
vLLM 服务 + 13 档固定 trace 压测
```

## Repository layout

```text
MedAdapt-Serve/
├── data/       # CMB 下载、检查、去重、SFT 构造、manifest
├── model/      # 权重下载、text-only 加载、LoRA parity
├── train/      # LoRA/QLoRA SFT、全量冒烟、训练守护
├── eval/       # 双协议评测、跨题库、通用保持、统计报告
├── serving/    # vLLM/SGLang 服务、量化导出、压测、监控
├── tests/      # 单元测试
├── optional/   # RLVR、自研 INT8、toy engine（扩展项，未纳入主线）
├── docs/       # 复现指南、结果口径、资产边界
├── out/        # 本地 checkpoint、评测与压测输出；GitHub 版本默认忽略
└── pyproject.toml
```

## Reproduction

### 1. Environment

Python 3.10+、PyTorch 2.8.0+cu129、transformers 5.15.0、PEFT 0.20.0、vLLM 0.19.0（版本锁定理由见 pyproject.toml 与 requirements-*.txt；训练、服务、量化分环境安装）。

### 2. Smoke test

```bash
python -m unittest discover -s tests -v
```

### 3. Data pipeline

```bash
python -m data.fetch_cmb
python -m data.inspect_cmb
python -m data.dedup_splits
python -m data.build_answer_sft
# 无 torch 环境必须显式 --char-fallback（字符近似口径）；有 tokenizer 的环境省略该参数
python -m serving.workload --output out/bench/workload.jsonl --per-bucket 10 --char-fallback
```

### 4. Training

```bash
python -m model.download --model-id Qwen/Qwen3.5-9B-Base

torchrun --standalone --nproc_per_node=4 -m train.train_lora \
  --model out/models/Qwen--Qwen3.5-9B-Base \
  --train-data data/processed/sft/train.jsonl \
  --eval-data data/processed/sft/val.jsonl \
  --output-dir out/checkpoints/lora \
  --max-length 4096
```

显存或量化内核不稳定时加 `--qlora`。训练异常中断时 watchdog 会用最新 checkpoint 自动续训；手动续训用 `--resume-from-checkpoint out/checkpoints/lora/checkpoint-<N>`。

### 5. Evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m eval.eval_cmb_logprob \
  --model out/models/Qwen--Qwen3.5-9B-Base \
  --data data/processed/cmb/val.jsonl \
  --output out/eval/base_val_logprob.json
```

### 6. Serving and benchmark

```bash
python -m serving.serve_vllm --model <merged_model_path> --host 127.0.0.1 --port 8000
```

量化导出（serving/quant_export.py）、压测（serving/bench.py）与结果汇总（serving/analyze_bench.py）的完整命令见 [docs/REPRODUCE.md](docs/REPRODUCE.md)。

CMB 原始数据、模型权重、checkpoint 和压测请求不随仓库发布；数据、权重与输出的发布边界见 [docs/ASSET_POLICY.md](docs/ASSET_POLICY.md)。

## Scientific scope

- 官方 CMB test 不含答案，主评测为 train 去重后训练完全未见的 5,000 条 held-out；跨题库 CMExam（题源与 CMB 不同）用于检验提升不是背题。
- 双协议评测的理由：自由生成 prompt 下 base 模型会无视指令、在 32 token 内截断于题干复述，得到低于随机的假象；改用 structured outputs 强制 JSON 输出后该问题消除。
- 服务正确性按同权重口径验证（vLLM 引擎 vs transformers 教师强制逐 token 对照）；跨权重对照只用于量化敏感性分析，不作为正确性证据。
- 量化结论绑定具体版本（vLLM 0.19.0、gptqmodel、驱动版本）与单机 4×3090 环境，跨环境外推需重新校验。

## Limitations

- answer-only 训练，不生成解析；医疗 CoT 推理是后续扩展
- val 280 条量级小，显著性结论以 held-out 与跨题库为准
- 医疗选择题准确率不等于诊疗能力，模型输出不构成医学建议
- 4×3090 单机环境的显存、功耗与吞吐结论不能直接外推到其他硬件
- 仓库不含 CMB 原始数据、模型权重、checkpoint 与压测请求

## License and data

CMB 数据与 Qwen 系列权重需遵守各自许可证。本仓库默认只提交源码、配置、测试和文档，不提交原始数据、处理后数据、模型权重和压测请求；公开发布前请按 [docs/ASSET_POLICY.md](docs/ASSET_POLICY.md) 审核数据来源、许可证和模型权重。
