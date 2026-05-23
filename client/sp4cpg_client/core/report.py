
from __future__ import annotations

from pathlib import Path
from html import escape
from datetime import datetime
from typing import Any


def export_html_report(result: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_rows = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>"
        for k, v in getattr(result, "metrics", {}).items()
    ) or "<tr><td colspan='2'>暂无指标；若使用 fallback 或纯预测模式，可能不产生完整评估指标。</td></tr>"
    artifact_rows = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>"
        for k, v in getattr(result, "artifacts", {}).items()
    ) or "<tr><td colspan='2'>暂无输出文件</td></tr>"
    step_rows = "".join(
        f"<tr><td>{escape(s.name)}</td><td>{s.returncode}</td><td>{escape(s.cwd)}</td><td><code>{escape(' '.join(map(str, s.command)))}</code></td></tr>"
        for s in getattr(result, "steps", [])
    ) or "<tr><td colspan='4'>暂无步骤记录</td></tr>"
    status = "成功" if getattr(result, "ok", False) else "失败"
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>HGCN + MEDE 检测报告</title>
<style>
body {{ margin:0; font-family:'Microsoft YaHei UI', Segoe UI, Arial; background:#f6f8fc; color:#172033; }}
.header {{ padding:34px 46px; background:linear-gradient(120deg,#eaf2ff,#ffffff 52%,#f3edff); border-bottom:1px solid #dbe4f0; }}
.header h1 {{ margin:0; font-size:31px; color:#0f172a; }}
.header p {{ color:#64748b; }}
.wrap {{ padding:28px 46px; }}
.card {{ background:#ffffff; border:1px solid #dbe4f0; border-radius:18px; padding:22px; margin-bottom:22px; box-shadow:0 10px 28px rgba(15,23,42,.06); }}
h2 {{ color:#1d4ed8; font-size:20px; margin-top:0; }}
table {{ width:100%; border-collapse: collapse; margin-top:12px; }}
th, td {{ border-bottom:1px solid #edf2f7; padding:11px; text-align:left; vertical-align:top; }}
th {{ color:#334155; background:#f1f5f9; }}
.badge {{ display:inline-block; padding:7px 14px; border-radius:999px; background:{'#16a34a' if getattr(result,'ok',False) else '#dc2626'}; color:white; font-weight:800; }}
code {{ color:#1d4ed8; word-break:break-all; }}
.note {{ color:#64748b; line-height:1.65; }}
</style>
</head>
<body>
<div class="header">
  <h1>HGCN + MEDE 漏洞检测报告</h1>
  <p>生成时间：{escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
  <span class="badge">{status}</span>
</div>
<div class="wrap">
  <div class="card"><h2>一、任务概况</h2>
    <table><tr><th>字段</th><th>值</th></tr>
      <tr><td>任务模式</td><td>{escape(str(getattr(result, 'mode', '--')))}</td></tr>
      <tr><td>输出目录</td><td>{escape(str(getattr(result, 'output_dir', '--')))}</td></tr>
      <tr><td>任务消息</td><td>{escape(str(getattr(result, 'message', '--')))}</td></tr>
    </table>
    <p class="note">报告汇总客户端执行步骤和输出文件。若报告中出现 fallback 字段，表示当前机器缺少真实 HGCN 或 MEDE 运行环境，客户端使用保底扫描结果完成流程演示。</p>
  </div>
  <div class="card"><h2>二、评估指标</h2><table><tr><th>指标</th><th>值</th></tr>{metrics_rows}</table></div>
  <div class="card"><h2>三、输出文件</h2><table><tr><th>类型</th><th>路径</th></tr>{artifact_rows}</table></div>
  <div class="card"><h2>四、执行步骤</h2><table><tr><th>步骤</th><th>返回码</th><th>工作目录</th><th>命令</th></tr>{step_rows}</table></div>
</div>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
