# MedAdapt-Serve Demo

离线证据模式（无需模型或 GPU）：

```bash
python demo/app.py
```

打开 `http://127.0.0.1:7863`，可查看项目内真实的 held-out、CMExam、GSM8K 评测汇总与 13 档固定 trace 压测 JSON。

接入真实 vLLM 服务：

```bash
python -m serving.serve_vllm --model <gptq_or_bf16_path> --host 127.0.0.1 --port 8000 --served-model-name medadapt-gptq
python demo/app.py --api-base http://127.0.0.1:8000/v1 --model medadapt-gptq
```

实时区会用 JSON Schema 强制返回 `{"answer":"A-E"}`，同时展示服务端 `usage` 和端到端耗时。该演示仅用于医疗选择题工程研究，不提供个人诊疗建议。
