# proj_3 实验结果与统计口径

以下数字来自 4×RTX 3090 实验产物和 README 记录。out/ 被 .gitignore 排除，GitHub 仓库不默认携带这些大文件；复现实验后应使用同名路径或在 release 中单独提供校验过的 artifact。

## 1. 主结果

| 评测 | Base | LoRA / GPTQ | 变化 | 解释 |
|---|---:|---:|---:|---|
| CMB val，280 条 logprob | 70.71% | LoRA 72.50% | +1.79 pt | 开发集规模较小 |
| CMB held-out，5,000 条 | 76.94% | LoRA 81.68% | **+4.74 pt** | SFT 未见样本，主领域结果 |
| CMExam，6,145 条 | 78.54% | LoRA 81.16% | **+2.62 pt** | 剔除 666 条 exact canonical overlap |
| GSM8K test，1,319 条 | 86.43% | LoRA 85.52% | −0.91 pt | 置信区间包含 0，未观察到显著下降 |
| CMB val，merged BF16 | 71.07% | GPTQ 72.86% | +1.79 pt | 量化对照，不解释为量化带来增益 |

held-out 5,000 条上的 LoRA 提升还报告了 bootstrap 95% CI [+3.82, +5.66] 和 McNemar p=7.2e-25；CMExam 提升的 bootstrap 95% CI 为 [+1.81, +3.43]，McNemar p=2.9e-10。

## 2. 训练与模型资产

- Base：Qwen/Qwen3.5-9B-Base。
- SFT：80,000 条 answer-only 数据，最大长度 4096。
- 训练：4×RTX 3090、FSDP、bf16、梯度累积，正式 checkpoint 为 checkpoint-1250。
- LoRA：可训练参数约 43.28M，占总参数约 0.48%。
- 全量 SFT smoke：100/100 step 完成，峰值 allocated 16.95 GiB、reserved 21.3 GiB。

## 3. 服务与压测

- 格式：GPTQ W4A16，group size 128，CMB train 前 512 条作为校准语料。
- 固定 trace：输入 256/1024/3072 token，输出 32/128/512 token，并发 1/4/8/16，包含无复用、固定前缀复用和长前缀复用。
- 13 档 greedy/seeded 压测请求失败数均为 0。
- 单并发生成吞吐：GPTQ 110.8 token/s，BF16 45.8 token/s。
- GPTQ 同权重服务对照：280 条中 271 条生成 token 序列一致；教师强制 logprob 差异 mean 约 0.0016、max 约 0.0235。

同权重服务对照与跨权重对照必须分开解释。跨权重 GPTQ-vs-BF16 的差异同时包含量化误差，不能作为 vLLM 服务正确性证据。

## 4. 证据文件

在本地实验目录中，推荐引用以下文件：

    out/manifests/cmb_dedup.json
    out/manifests/heldout_build.json
    out/manifests/cmexam_build.json
    out/manifests/lora_parity.json
    out/eval/base_heldout5000_logprob.json
    out/eval/lora_heldout5000_logprob_ckpt1250.json
    out/eval/cross_domain_cmexam_base_structured.json
    out/eval/cross_domain_cmexam_lora_structured.json
    out/eval/general_gsm8k_base.json
    out/eval/general_gsm8k_lora.json
    out/eval/cross_check_gptq_same_weights.json
    out/bench/bench_*.json

建议对外发布前增加一个 out/eval/INDEX.json，明确每个结果的 valid、数据 hash、Base、adapter/merged model、git commit、命令和环境指纹，避免旧实验结果与最终结果混淆。

## 5. 结果边界

1. 当前污染检查是 exact canonical-key 检查，不是近重复检测。
2. held-out 与训练数据来自同一 CMB 来源，证明的是训练未见表现，不是完全独立外部泛化。
3. CMExam 与 CMB 都属于医考相关题源，跨题库提升不应表述为通用医疗知识全面提升。
4. CMB val 只有 280 条，适合调试与早期门控，不适合作为唯一主结论。
5. 当前 GitHub 仓库不包含 Base、merged BF16、GPTQ 权重和处理后数据；这些资产需要单独下载或通过 release 提供。
