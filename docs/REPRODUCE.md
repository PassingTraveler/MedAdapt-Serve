# Reproduce proj_3

本文档给出一条可审计的复现实验路径。主线假设为 Linux、Python 3.10–3.13、4×RTX 3090；无 GPU 环境可以运行数据管线、单测和命令构造，但不能复现训练与服务性能数字。

## 1. 仓库与环境

    git clone <your-repository-url>
    cd proj_3
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements-smoke.txt

无 GPU 的第一步验证：

    PYTHONPATH=$PWD python -m unittest discover -s tests -v
    ruff check .
    python -m compileall -q data model train eval serving optional scripts tests

训练、服务和量化使用分离环境，避免 transformers、torch 和 vllm 的版本约束相互冲突：

    # 训练/评测环境
    pip install -r requirements-train.txt

    # vLLM 服务环境
    pip install -r requirements-serve.txt

    # 只有需要重新导出 GPTQ 时安装
    pip install -r requirements-quant.txt

实际复现时还应记录 Python、PyTorch、CUDA、驱动、NCCL、GPU 型号、完整 pip freeze 和仓库 commit。

## 2. 数据管线

    python -m data.fetch_cmb
    python -m data.inspect_cmb
    python -m data.dedup_splits
    python -m data.build_answer_sft
    python -m data.build_answer_sft +      --input data/processed/cmb/val.jsonl +      --output data/processed/sft/val.jsonl +      --max-records 280 +      --no-shuffle
    python -m data.build_heldout

跨题库和通用保持评测数据：

    python -m data.build_cmexam
    python -m data.build_general
    python -m data.build_manifest

数据构建脚本会在 out/manifests/ 写入数量、去重、来源和输出统计。CMB 官方 test 不含公开答案，不能直接用于本地准确率评测；训练未见 held-out 和 CMExam 用于可计算的准确率对照。

## 3. 模型资产

GitHub 不包含模型权重。建议固定 Hugging Face revision，而不是使用随时间变化的 main：

    python -m model.download +      --model-id Qwen/Qwen3.5-9B-Base +      --revision <base-model-commit> +      --output-dir out/models/Qwen--Qwen3.5-9B-Base

下载完成后保存 download_manifest.json，并在实验 manifest 中记录每个文件的大小和 SHA256。

## 4. LoRA 训练

正式训练前建议先跑 500–1000 step 的 early-learning gate，确认验证集 logprob 相对 Base 有可观察变化，再继续完整训练。

    CUDA_VISIBLE_DEVICES=0,1,2,3 +    torchrun --standalone --nproc_per_node=4 +      -m train.train_lora +      --model out/models/Qwen--Qwen3.5-9B-Base +      --train-data data/processed/sft/train.jsonl +      --eval-data data/processed/sft/val.jsonl +      --output-dir out/checkpoints/lora +      --max-length 4096

中断后从 checkpoint 恢复：

    python -m train.train_lora +      --model out/models/Qwen--Qwen3.5-9B-Base +      --resume-from-checkpoint out/checkpoints/lora/checkpoint-<step> +      --train-data data/processed/sft/train.jsonl +      --eval-data data/processed/sft/val.jsonl +      --output-dir out/checkpoints/lora

LoRA 注入与 merge parity：

    python -m model.lora_parity +      --model out/models/Qwen--Qwen3.5-9B-Base +      --output out/manifests/lora_parity.json

## 5. 评测

Base 的候选答案 logprob：

    python -m eval.eval_cmb_logprob +      --model out/models/Qwen--Qwen3.5-9B-Base +      --data data/processed/cmb/heldout_5000.jsonl +      --output out/eval/base_heldout5000_logprob.json

LoRA 评测必须显式提供 adapter，避免结果文件无法追溯模型来源：

    python -m eval.eval_cmb_logprob +      --model out/models/Qwen--Qwen3.5-9B-Base +      --peft-adapter out/checkpoints/lora/checkpoint-1250 +      --data data/processed/cmb/heldout_5000.jsonl +      --output out/eval/lora_heldout5000_logprob_ckpt1250.json

生成式跨题库和通用保持评测需要先启动服务，具体参数见 eval/eval_cross_domain.py 与 eval/eval_general.py。

## 6. Merge、GPTQ 与 vLLM

先把 adapter merge 回 Base：

    python scripts/merge_lora.py +      --base out/models/Qwen--Qwen3.5-9B-Base +      --adapter out/checkpoints/lora/checkpoint-1250 +      --output out/models/Qwen--Qwen3.5-9B-Base-LoRA-ckpt1250

GPTQ 导出只在单独的量化环境执行：

    python -m serving.quant_export +      --model out/models/Qwen--Qwen3.5-9B-Base-LoRA-ckpt1250 +      --output-dir out/models/Qwen--Qwen3.5-9B-Base-LoRA-ckpt1250-GPTQ-W4A16 +      --bits 4 +      --group-size 128 +      --vllm-load-check +      --vllm-python <vllm-python>

启动服务：

    python -m serving.serve_vllm +      --model out/models/Qwen--Qwen3.5-9B-Base-LoRA-ckpt1250-GPTQ-W4A16 +      --quantization gptq +      --max-model-len 4096 +      --tensor-parallel-size 1

默认只监听 127.0.0.1。对外提供服务前必须显式配置 host、鉴权和网络隔离。

## 7. 固定 trace 压测

有本地 tokenizer 时按真实 token 计量；无 tokenizer 的字符 fallback 只用于命令冒烟，不能用于最终性能结论。

    python -m serving.workload +      --output out/bench/workload.jsonl +      --per-bucket 100

    python -m serving.bench +      --workload out/bench/workload.jsonl +      --base-url http://127.0.0.1:8000/v1 +      --model Qwen3.5-9B-GPTQ-W4A16 +      --concurrency 1 +      --output out/bench/bench_gptq_c1.json

最终报告应同时保存 trace、服务参数、失败数、TTFT、TPOT、吞吐、P50/P99、显存、功耗、模型 hash 和环境信息。

## 8. 复现边界

- 当前协议完成的是精确 canonical-key 去重，不等同于 fuzzy/embedding 近重复检测。
- held-out 来自 CMB train 中未进入 SFT 的样本，适合验证训练未见表现，但不是独立来源的外部测试集。
- CMExam 与 CMB 均属于中文医考相关数据，跨题库结果应称为跨题源验证，不应夸大为完全不同领域泛化。
- CMB val 仅 280 条，适合作为开发监控，不应单独支撑强结论。
