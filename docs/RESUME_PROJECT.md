# 简历项目卡

## 推荐标题

Qwen3.5-9B 中文医疗适配与可审计推理部署

## 时间

2026.08–至今

## 项目简介

面向中文医疗选择题场景，基于 Qwen3.5-9B Base 完成数据治理、LoRA 微调、跨题库评测及 GPTQ/vLLM 部署，重点验证领域能力提升、通用能力保持与推理效率之间的权衡。

## 个人工作

数据方面，完成 CMB 数据的 split 内/跨 split 精确去重、选项重排和答案映射，构建 80,000 条 answer-only SFT 数据、5,000 条训练未见 held-out 数据，并对 CMExam 剔除 666 条 canonical-key 重叠样本。训练方面，针对 Qwen3.5 混合结构实现 text-only 加载与 PEFT LoRA 适配，结合 FSDP、bf16 和梯度累积，在 4×RTX 3090 上完成 9B 模型训练。评测方面，搭建生成式答案匹配与候选答案 logprob 双协议，LoRA 在 held-out 集上较 Base 提升 4.74 个百分点，在 CMExam 上提升 2.62 个百分点，GSM8K 下降 0.91 个百分点且无显著差异。部署方面，完成 GPTQ W4A16 导出及 vLLM 服务验证，构建 13 档固定 trace 压测，单并发生成吞吐达到 110.8 token/s，较 BF16 提升约 2.4 倍，压测请求零失败。

## 面试关键词

Qwen3.5-9B、PEFT/LoRA、FSDP、answer-only SFT、数据去重与污染审计、logprob 评测、跨题库评测、GSM8K 保持、GPTQ W4A16、vLLM、TTFT/TPOT/吞吐、固定 trace、可复现性。
