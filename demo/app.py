#!/usr/bin/env python
"""MedAdapt-Serve 的本地服务交互与压测证据 Demo。

默认只展示仓库保存的评测/压测产物；传入 --api-base 后才会调用本机
OpenAI 兼容 vLLM 服务。该 Demo 仅用于工程展示，不提供医疗建议。
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "out" / "bench"
EVAL_DIR = ROOT / "out" / "eval"
CONFIG = {"api_base": None, "model": None}


def summary_from(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def load_evidence() -> dict:
    """从 canonical JSON 读取，不在 Demo 中重算或改写指标。"""
    bench = []
    for fmt in ("bf16", "gptq"):
        for concurrency in (1, 4, 8, 16):
            path = BENCH_DIR / f"bench_{fmt}_c{concurrency}_greedy.json"
            if not path.is_file():
                continue
            row = summary_from(path)
            bench.append({
                "format": "GPTQ W4A16" if fmt == "gptq" else "BF16",
                "concurrency": row["concurrency"],
                "throughput": round(row["completion_tokens_per_second"], 1),
                "ttft": round(row["ttft_ms"]["p50"], 1),
                "tpot": round(row["tpot_ms"]["p50"], 2),
                "requests": row["requests"],
                "failed": row["failed"],
            })
    metrics = {
        "heldout_base": summary_from(EVAL_DIR / "base_heldout5000_logprob.json")["accuracy"],
        "heldout_lora": summary_from(EVAL_DIR / "lora_heldout5000_logprob_ckpt1250.json")["accuracy"],
        "cmexam_base": summary_from(EVAL_DIR / "cross_domain_cmexam_base_structured.json")["accuracy"],
        "cmexam_lora": summary_from(EVAL_DIR / "cross_domain_cmexam_lora_structured.json")["accuracy"],
        "gsm8k_base": summary_from(EVAL_DIR / "general_gsm8k_base.json")["accuracy"],
        "gsm8k_lora": summary_from(EVAL_DIR / "general_gsm8k_lora.json")["accuracy"],
    }
    return {"bench": bench, "metrics": metrics}


EVIDENCE = load_evidence()


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MedAdapt-Serve · 工程 Demo</title>
<style>
:root{--ink:#21202e;--muted:#6b6979;--accent:#6c3ffc;--line:#e7e4f0;--warn:#fff4dc;--bg:#f8f7fc}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,"Microsoft YaHei",sans-serif}.shell{max-width:1140px;margin:auto;padding:34px 24px 54px}.eyebrow{color:var(--accent);font-size:12px;font-weight:800;letter-spacing:.13em;text-transform:uppercase}h1{font-size:clamp(29px,5vw,46px);line-height:1.08;letter-spacing:-.04em;margin:10px 0}.lead{font-size:17px;color:var(--muted);max-width:820px}.alert{margin:20px 0;padding:12px 14px;background:var(--warn);border-left:4px solid #ffae31;border-radius:8px;color:#73531f}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.metric,.card{background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:0 10px 26px rgba(39,23,87,.04)}.metric{padding:15px}.metric b{display:block;color:var(--accent);font-size:24px}.metric span,.sub{font-size:12px;color:var(--muted)}.grid{display:grid;grid-template-columns:1.06fr .94fr;gap:16px}.card{padding:20px}h2{font-size:20px;margin:0 0 3px}textarea,button{font:inherit}textarea{display:block;width:100%;min-height:155px;padding:12px;border:1px solid #d8d4e8;border-radius:11px;resize:vertical;line-height:1.5}button{border:0;border-radius:10px;background:var(--accent);color:#fff;font-weight:700;padding:10px 14px;margin-top:10px;cursor:pointer}button:disabled{opacity:.55;cursor:wait}.result{white-space:pre-wrap;min-height:130px;margin-top:12px;padding:13px;background:#f5f2ff;border-radius:12px;overflow:auto}.result.error{background:#fff0ec;color:#9d2e13}.status{display:inline-block;margin-top:9px;padding:3px 8px;border-radius:99px;background:#edf1ff;color:#5045b8;font-size:12px;font-weight:700}.governance{margin-top:16px}.steps{display:grid;gap:8px;margin-top:12px}.step{padding:10px 12px;border-radius:10px;background:#f7f9ff;border-left:3px solid #a895ff}.step b{display:inline-block;width:84px}.bench{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}.bench th,.bench td{text-align:right;padding:8px 6px;border-bottom:1px solid var(--line)}.bench th:first-child,.bench td:first-child{text-align:left}.good{color:#127348;font-weight:700}code{font-family:ui-monospace,Consolas,monospace}.command{padding:11px;background:#f3f1f9;border-radius:10px;overflow:auto;color:#4c4767}footer{margin-top:20px;color:var(--muted);font-size:12px}@media(max-width:760px){.shell{padding:24px 15px}.metrics{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.bench{font-size:12px}}
</style></head><body><main class="shell"><div class="eyebrow">MedAdapt-Serve · auditable serving demo</div><h1>选择题适配、可审计评测、<br>量化服务在一个闭环里。</h1><p class="lead">页面展示真实评测记录、13 档固定 trace 的服务证据，并可选择接入本机 vLLM 做实时结构化输出。未配置服务时，页面不会编造模型回答。</p><div class="alert">医疗安全边界：本 Demo 只用于模型工程与医学选择题研究展示，不提供个人诊疗、处方或风险判断。</div>
<section class="metrics"><div class="metric"><b id="heldout">—</b><span>held-out 5,000：LoRA / Base</span></div><div class="metric"><b id="cmexam">—</b><span>CMExam 6,145：LoRA / Base</span></div><div class="metric"><b id="gsm">—</b><span>GSM8K：LoRA / Base</span></div><div class="metric"><b>110.8 tok/s</b><span>GPTQ 单并发真实 token 吞吐</span></div></section>
<section class="grid"><article class="card"><h2>实时 vLLM 结构化输出</h2><p class="sub">只有启动后端并用 <code>--api-base</code> 配置本页时才调用模型。默认题目为合成接口检查题；可替换为非个体化选择题。</p><textarea id="prompt">【接口演示题，不构成医疗建议】&#10;某研究记录中，只有一个候选标签与给定规则匹配。请选择最符合规则的选项。&#10;A. 标签一&#10;B. 标签二&#10;C. 标签三&#10;D. 标签四&#10;E. 标签五&#10;&#10;请只返回 JSON：{"answer":"A-E 之间的一个字母"}</textarea><button id="ask">请求本机服务</button><div class="status" id="status">检测中…</div><div class="result" id="result">尚未发送请求。</div></article>
<article class="card"><h2>数据治理与评测协议</h2><p class="sub">这些是项目强说服力的地方：每一步都有可追溯文件，而不是只展示最终准确率。</p><div class="steps"><div class="step"><b>80,000 条</b>CMB answer-only SFT；split 内/跨 split 去重、选项重排与答案映射。</div><div class="step"><b>5,000 条</b>训练未见 held-out；生成式匹配与候选 logprob 双协议。</div><div class="step"><b>666 条</b>CMExam canonical key 重叠剔除，剩余 6,145 条跨题库测试。</div><div class="step"><b>保持检查</b>GSM8K 报告变化与显著性，不将领域增益包装成通用能力提升。</div></div><h2 style="margin-top:23px">启动真实服务</h2><div class="command"><code>python -m serving.serve_vllm --model &lt;gptq_or_bf16_path&gt; --host 127.0.0.1 --port 8000 --served-model-name medadapt-gptq<br>python demo/app.py --api-base http://127.0.0.1:8000/v1 --model medadapt-gptq</code></div></article></section>
<section class="card governance"><h2>固定 trace 压测回放</h2><p class="sub">直接读取 <code>out/bench/bench_*.json</code> 的 summary；TTFT/TPOT 以服务端 usage 返回的真实 completion token 计量。所有显示 trace 均为 0 失败。</p><table class="bench"><thead><tr><th>格式</th><th>并发</th><th>吞吐 tok/s</th><th>p50 TTFT ms</th><th>p50 TPOT ms</th><th>请求数</th><th>失败</th></tr></thead><tbody id="bench"></tbody></table></section><footer>离线证据与在线推理明确分离：页面指标可追溯，实时模型返回只在本机服务已配置时显示。</footer></main>
<script>
const $=s=>document.querySelector(s);const pct=x=>`${(x*100).toFixed(2)}%`;
async function boot(){const x=await(await fetch('/api/evidence')).json(),m=x.metrics;$('#heldout').textContent=`${pct(m.heldout_lora)} / ${pct(m.heldout_base)}`;$('#cmexam').textContent=`${pct(m.cmexam_lora)} / ${pct(m.cmexam_base)}`;$('#gsm').textContent=`${pct(m.gsm8k_lora)} / ${pct(m.gsm8k_base)}`;$('#bench').innerHTML=x.bench.map(r=>`<tr><td>${r.format}</td><td>${r.concurrency}</td><td>${r.throughput}</td><td>${r.ttft}</td><td>${r.tpot}</td><td>${r.requests}</td><td class="${r.failed===0?'good':''}">${r.failed}</td></tr>`).join('');const c=await(await fetch('/api/config')).json();$('#status').textContent=c.enabled?`已配置：${c.model} @ ${c.api_base}`:'离线证据模式：未配置 vLLM'}
async function ask(){const btn=$('#ask'),out=$('#result');btn.disabled=true;out.className='result';out.textContent='请求已发送，等待本机 vLLM 返回…';try{const r=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({prompt:$('#prompt').value})});const x=await r.json();if(!r.ok)throw new Error(x.error||'服务请求失败');out.textContent=`结构化回答：${x.answer||'(未解析)'}\n\n原始 content：\n${x.content}\n\nE2E：${x.elapsed_ms} ms\nusage：${JSON.stringify(x.usage||{})}`}catch(e){out.className='result error';out.textContent=`未调用到实时模型：${e.message}\n\n先启动 vLLM，再用 --api-base 与 --model 启动本 Demo。离线证据仍可正常浏览。`}finally{btn.disabled=false}}
$('#ask').onclick=ask;boot();
</script></body></html>"""


def call_vllm(prompt: str) -> dict:
    if not CONFIG["api_base"] or not CONFIG["model"]:
        raise RuntimeError("未配置 --api-base 与 --model")
    base = str(CONFIG["api_base"]).rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    schema = {
        "name": "medical_multiple_choice_answer",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string", "pattern": "^[A-E]$"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        "strict": True,
    }
    payload = {
        "model": CONFIG["model"],
        "messages": [
            {"role": "system", "content": "你只做选择题标签输出，不提供诊疗建议。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 32,
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    started = time.perf_counter()
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 vLLM: {exc.reason}") from exc
    content = raw["choices"][0]["message"].get("content", "")
    try:
        answer = json.loads(content).get("answer")
    except json.JSONDecodeError:
        answer = None
    return {"content": content, "answer": answer, "usage": raw.get("usage"), "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        return

    def _send(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        self._send(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            return self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/api/evidence":
            return self._json(EVIDENCE)
        if path == "/api/config":
            return self._json({"enabled": bool(CONFIG["api_base"] and CONFIG["model"]), **CONFIG})
        if path == "/health":
            return self._json({"ok": True, "api_configured": bool(CONFIG["api_base"] and CONFIG["model"])})
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            prompt = str(json.loads(self.rfile.read(length) or b"{}").get("prompt", "")).strip()
            if not prompt:
                raise ValueError("题目不能为空")
            if len(prompt) > 12000:
                raise ValueError("题目过长（上限 12,000 字符）")
            self._json(call_vllm(prompt))
        except (ValueError, RuntimeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)


def main() -> None:
    parser = argparse.ArgumentParser(description="MedAdapt-Serve demo")
    parser.add_argument("--host", default="127.0.0.1", help="默认仅监听本机")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--api-base", help="本机 OpenAI 兼容 vLLM 地址，如 http://127.0.0.1:8000/v1")
    parser.add_argument("--model", help="vLLM 的 served-model-name")
    args = parser.parse_args()
    CONFIG.update({"api_base": args.api_base, "model": args.model})
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"MedAdapt-Serve demo: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
