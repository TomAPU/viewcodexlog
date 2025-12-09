#!/usr/bin/env python3
"""
Simple viewer for Codex CLI JSONL logs.

Usage:
    python3 viewcodexlog.py -l <logfile.jsonl> -p <port>
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Iterable, List, Optional


@dataclass
class Entry:
    timestamp: str
    label: str
    body_html: str
    css_class: str
    raw_type: str
    lineno: int
    extra_classes: List[str] = field(default_factory=list)


@dataclass
class RunCodeUpload:
    index: int
    timestamp: str
    lineno: int
    code: str
    flags: str


TARGET_RUN_CODE_FN = "mcp__kernelmcp__vm_compile_c_and_upload"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Codex JSONL logs as HTML.")
    parser.add_argument(
        "-l",
        "--log",
        required=True,
        help="Path to the JSONL conversation log.",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server (default: 8000).",
    )
    return parser.parse_args()


def load_entries(path: Path) -> List[Entry]:
    entries: List[Entry] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                entries.append(
                    Entry(
                        timestamp="n/a",
                        label="Malformed JSON",
                        body_html=f"<pre>{html.escape(str(exc))}</pre>",
                        css_class="entry-error",
                        raw_type="error",
                        lineno=lineno,
                    )
                )
                continue
            entry = convert_record(record, lineno)
            if entry:
                entries.append(entry)
    return entries


def convert_record(record: dict, lineno: int) -> Optional[Entry]:
    rectype = record.get("type", "unknown")
    timestamp = record.get("timestamp", "unknown")
    payload = record.get("payload", {})

    if rectype == "session_meta":
        body = format_payload(payload, collapsed=True)
        return Entry(
            timestamp=timestamp,
            label="Session metadata",
            body_html=body,
            css_class="entry-system",
            raw_type=rectype,
            lineno=lineno,
        )

    if rectype == "turn_context":
        body = format_payload(payload, collapsed=True)
        return Entry(
            timestamp=timestamp,
            label="Turn context",
            body_html=body,
            css_class="entry-system",
            raw_type=rectype,
            lineno=lineno,
            extra_classes=["collapsible-meta"],
        )

    if rectype == "response_item":
        return convert_response_item(record, lineno)

    if rectype == "event_msg":
        return convert_event_msg(record, lineno)

    # Unknown type, display raw payload for debugging.
    fallback = format_payload(payload)
    return Entry(
        timestamp=timestamp,
        label=f"Unhandled type: {rectype}",
        body_html=fallback,
        css_class="entry-system",
        raw_type=rectype,
        lineno=lineno,
    )


def convert_response_item(record: dict, lineno: int) -> Optional[Entry]:
    payload = record.get("payload") or {}
    subtype = payload.get("type")
    timestamp = record.get("timestamp", "unknown")
    role = payload.get("role", "n/a")

    if subtype == "message":
        texts = extract_text_chunks(payload.get("content") or [])
        if not texts:
            return None
        text_html = "<hr>".join(format_text_block(t) for t in texts)
        css = "entry-user" if role == "user" else "entry-assistant"
        return Entry(
            timestamp=timestamp,
            label=f"Message · {role}",
            body_html=text_html,
            css_class=css,
            raw_type="response_item/message",
            lineno=lineno,
        )

    if subtype == "function_call":
        name = payload.get("name", "unknown")
        args = payload.get("arguments") or ""
        call_id = payload.get("call_id", "n/a")
        parsed_args = try_parse_json(args)
        plan_html = render_plan_board(
            parsed_args) if name == "update_plan" else None
        args_html = ""
        if parsed_args is not None:
            args_html = render_structured_data(parsed_args)
        elif args:
            args_html = format_pre(args)
        body = (
            f"<div><strong>Call:</strong> {html.escape(name)}</div>"
            f"<div><strong>call_id:</strong> {html.escape(call_id)}</div>"
        )
        if plan_html:
            body += plan_html
        if args_html:
            body += args_html
        return Entry(
            timestamp=timestamp,
            label="Function call",
            body_html=body,
            css_class="entry-tool",
            raw_type="response_item/function_call",
            lineno=lineno,
        )

    if subtype == "function_call_output":
        call_id = payload.get("call_id", "n/a")
        output = payload.get("output")
        parsed_output = try_parse_json(output)
        if parsed_output is not None:
            output_html = render_structured_data(parsed_output)
        elif output is None:
            output_html = "<em>no output</em>"
        else:
            output_html = render_scalar(output)
        body = f"<div><strong>call_id:</strong> {html.escape(call_id)}</div>{output_html}"
        return Entry(
            timestamp=timestamp,
            label="Function output",
            body_html=body,
            css_class="entry-tool",
            raw_type="response_item/function_call_output",
            lineno=lineno,
        )

    if subtype == "reasoning":
        summary = payload.get("summary") or []
        if summary:
            summary_html = "<ul>" + "".join(
                f"<li>{render_reasoning_summary_item(item)}</li>" for item in summary
            ) + "</ul>"
        else:
            summary_html = "<em>No public summary (content encrypted)</em>"
        return Entry(
            timestamp=timestamp,
            label="Reasoning note",
            body_html=summary_html,
            css_class="entry-assistant",
            raw_type="response_item/reasoning",
            lineno=lineno,
            extra_classes=["collapsible-meta"],
        )

    return Entry(
        timestamp=timestamp,
        label=f"Response item ({subtype or 'unknown'})",
        body_html=format_payload(payload),
        css_class="entry-system",
        raw_type="response_item/unknown",
        lineno=lineno,
    )


def convert_event_msg(record: dict, lineno: int) -> Entry:
    payload = record.get("payload") or {}
    subtype = payload.get("type")
    timestamp = record.get("timestamp", "unknown")

    if subtype in {"user_message", "agent_message"}:
        message = payload.get("message", "")
        kind = payload.get("kind", "plain")
        css = "entry-user" if subtype == "user_message" else "entry-assistant"
        body = (
            f"<div><strong>Kind:</strong> {html.escape(kind)}</div>"
            f"{format_pre(message)}"
        )
        return Entry(
            timestamp=timestamp,
            label=f"Event · {subtype}",
            body_html=body,
            css_class=css,
            raw_type=f"event_msg/{subtype}",
            lineno=lineno,
        )

    if subtype == "token_count":
        info = payload.get("info") or {}
        body = format_payload(info, collapsed=True)
        return Entry(
            timestamp=timestamp,
            label="Token usage",
            body_html=body,
            css_class="entry-metric",
            raw_type="event_msg/token_count",
            lineno=lineno,
            extra_classes=["collapsible-meta"],
        )

    return Entry(
        timestamp=timestamp,
        label=f"Event ({subtype or 'unknown'})",
        body_html=format_payload(payload),
        css_class="entry-system",
        raw_type=f"event_msg/{subtype or 'unknown'}",
        lineno=lineno,
    )


def extract_text_chunks(content_items: Iterable[dict]) -> List[str]:
    texts: List[str] = []
    for chunk in content_items:
        if not isinstance(chunk, dict):
            continue
        if chunk.get("type") in {"input_text", "output_text"}:
            text = chunk.get("text")
            if text:
                texts.append(text)
    return texts


def format_text_block(text: str) -> str:
    escaped = html.escape(text)
    return escaped.replace("\n", "<br>")


def render_reasoning_summary_item(item: object) -> str:
    if isinstance(item, dict):
        kind = item.get("type")
        text = item.get("text")
        if isinstance(kind, str) and text is not None:
            text_value = text if isinstance(text, str) else str(text)
            text_html = format_text_block(text_value)
            kind_html = html.escape(kind)
            return f"<strong>{kind_html}</strong>: {text_html}"
    return html.escape(str(item))


def format_pre(text: str) -> str:
    if text is None:
        return ""
    return f"<pre>{html.escape(str(text))}</pre>"


def format_payload(payload: dict, collapsed: bool = False) -> str:
    pretty = json.dumps(payload, indent=2, ensure_ascii=False)
    escaped = html.escape(pretty)
    pre = f"<pre>{escaped}</pre>"
    if not collapsed:
        return pre
    return f"<details><summary>Show payload</summary>{pre}</details>"


def try_parse_json(value: object) -> Optional[object]:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def render_structured_data(data: object) -> str:
    if isinstance(data, dict):
        if data.get("type") == "code":
            return render_code_block(data)
        rows = "".join(
            f"<tr><th>{html.escape(str(key))}</th><td>{render_structured_data(value)}</td></tr>"
            for key, value in data.items()
        )
        return f'<table class="kv-table">{rows}</table>'
    if isinstance(data, list):
        items = "".join(
            f"<li>{render_structured_data(item)}</li>" for item in data)
        return f'<ul class="list-nested">{items}</ul>'
    return render_scalar(data)


def render_scalar(value: object) -> str:
    if value is None:
        return "<em>null</em>"
    if isinstance(value, str):
        return format_pre(value) if "\n" in value else f"<span>{html.escape(value)}</span>"
    return f"<span>{html.escape(str(value))}</span>"


def render_code_block(node: dict) -> str:
    code = node.get("code") or node.get("content") or node.get("text") or ""
    code = str(code)
    language = node.get("language") or node.get(
        "lang") or node.get("programming_language")
    header = f'<div class="code-lang">{html.escape(language)}</div>' if language else ""
    return f'<div class="code-block">{header}<pre><code>{html.escape(code)}</code></pre></div>'


def render_plan_board(data: Optional[object]) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    plan = data.get("plan")
    if not isinstance(plan, list):
        return None
    explanation = data.get("explanation")
    items = []
    for item in plan:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "unknown"))
        status_class = status.lower().replace(" ", "-")
        step = html.escape(str(item.get("step", "")))
        items.append(
            f'<li><span class="status-chip status-{status_class}">{html.escape(status.replace("_", " "))}</span>'
            f"<span>{step}</span></li>"
        )
    if not items:
        return None
    expl_html = f"<p>{html.escape(str(explanation))}</p>" if explanation else ""
    return f'<section class="plan-board"><h4>Plan</h4>{expl_html}<ol>{"".join(items)}</ol></section>'


def render_diff(diff_text: str) -> str:
    if not diff_text:
        return "<pre class=\"diff-block\"><span class=\"diff-context\">(no diff)</span></pre>"
    formatted: List[str] = []
    for line in diff_text.splitlines():
        escaped = html.escape(line) or "&nbsp;"
        cls = "diff-context"
        if line.startswith("@@"):
            cls = "diff-hunk"
        elif line.startswith("+++ ") or line.startswith("--- "):
            cls = "diff-file"
        elif line.startswith("+") and not line.startswith("+++"):
            cls = "diff-add"
        elif line.startswith("-") and not line.startswith("---"):
            cls = "diff-del"
        formatted.append(f'<span class="{cls}">{escaped}</span>')
    return f'<pre class="diff-block">{"".join(formatted)}</pre>'


BASE_CSS = """
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      background: #f5f6f8;
      color: #1c1c1c;
    }
    header {
      padding: 1rem 2rem;
      background: #232f3e;
      color: white;
    }
    .header-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      flex-wrap: wrap;
    }
    .header-actions {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .container {
      padding: 1rem 2rem 3rem;
    }
    .entry {
      background: white;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      padding: 1rem;
      margin-bottom: 1rem;
      border-left: 4px solid transparent;
    }
    .entry header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 0.5rem;
      padding: 0;
      background: none;
      color: inherit;
    }
    .entry-system { border-color: #6c757d; }
    .entry-user { border-color: #007bff; }
    .entry-assistant { border-color: #6f42c1; }
    .entry-tool { border-color: #e36209; }
    .entry-metric { border-color: #198754; }
    .entry-error { border-color: #dc3545; }
    pre {
      background: #1e1e1e;
      color: #f8f8f2;
      padding: 0.75rem;
      overflow-x: auto;
      border-radius: 6px;
    }
    details summary {
      cursor: pointer;
      font-weight: 600;
      margin-bottom: 0.5rem;
    }
    hr {
      border: none;
      border-top: 1px solid #e5e5e5;
      margin: 0.75rem 0;
    }
    .kv-table {
      width: 100%;
      border-collapse: collapse;
      margin: 0.5rem 0;
    }
    .kv-table th,
    .kv-table td {
      padding: 0.35rem 0.5rem;
      border-bottom: 1px solid #e5e5e5;
      vertical-align: top;
    }
    .kv-table th {
      text-align: left;
      width: 180px;
      color: #495057;
      background: #f8f9fa;
    }
    .list-nested {
      margin: 0.25rem 0 0.25rem 1.25rem;
      padding-left: 1rem;
    }
    .list-nested li {
      margin-bottom: 0.35rem;
    }
    .plan-board {
      background: #f8f9fb;
      border-radius: 6px;
      padding: 0.75rem;
      margin: 0.5rem 0;
      border: 1px solid #e3e7ed;
    }
    .plan-board h4 {
      margin: 0 0 0.5rem;
    }
    .plan-board ol {
      margin: 0;
      padding-left: 1.25rem;
    }
    .status-chip {
      display: inline-block;
      padding: 0.1rem 0.6rem;
      border-radius: 999px;
      font-size: 0.8rem;
      margin-right: 0.5rem;
      text-transform: capitalize;
    }
    .status-chip.status-in_progress {
      background: #fff3cd;
      color: #7c5b07;
    }
    .status-chip.status-pending {
      background: #e9ecef;
      color: #495057;
    }
    .status-chip.status-completed {
      background: #d1e7dd;
      color: #0f5132;
    }
    .status-chip.status-error {
      background: #f8d7da;
      color: #842029;
    }
    .code-block {
      margin: 0.5rem 0;
    }
    .code-lang {
      font-size: 0.78rem;
      color: #6c757d;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.25rem;
    }
    .meta-toggle,
    .nav-button {
      border: none;
      background: #ffc107;
      color: #1c1c1c;
      padding: 0.5rem 1rem;
      border-radius: 999px;
      cursor: pointer;
      font-weight: 600;
      transition: background 0.2s ease;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .nav-button {
      background: #0d6efd;
      color: white;
    }
    .nav-button.secondary {
      background: #6c757d;
    }
    .meta-toggle:hover {
      background: #ffca2c;
    }
    .nav-button:hover {
      background: #0b5ed7;
    }
    body.meta-hidden .entry.collapsible-meta {
      display: none;
    }
    .panel {
      background: white;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      padding: 1rem;
      margin-bottom: 1rem;
    }
    .panel h2 {
      margin-top: 0;
    }
    .uploads-table {
      width: 100%;
      border-collapse: collapse;
    }
    .uploads-table th,
    .uploads-table td {
      border-bottom: 1px solid #e5e5e5;
      padding: 0.35rem 0.5rem;
      text-align: left;
    }
    .uploads-table th {
      background: #f8f9fa;
    }
    .diff-card {
      background: white;
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 1rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      border-left: 4px solid #0d6efd;
    }
    .diff-card h3 {
      margin-top: 0;
      margin-bottom: 0.5rem;
    }
    .diff-card pre {
      margin: 0;
    }
    .error-banner {
      background: #f8d7da;
      color: #842029;
      padding: 0.75rem 1rem;
      border-radius: 6px;
    }
    .diff-block {
      background: #0b0d12;
      color: #e6edf3;
      padding: 0.75rem;
      border-radius: 6px;
      font-family: "SFMono-Regular", Consolas, Menlo, monospace;
      font-size: 0.9rem;
      line-height: 1.35;
      overflow-x: auto;
      white-space: pre;
    }
    .diff-block span {
      display: block;
      padding: 0 0.35rem;
      border-radius: 4px;
    }
    .diff-add {
      background: rgba(46, 160, 67, 0.25);
      color: #7ee787;
    }
    .diff-del {
      background: rgba(248, 81, 73, 0.25);
      color: #ffaba8;
    }
    .diff-hunk {
      color: #79c0ff;
    }
    .diff-file {
      color: #ffa657;
    }
    .diff-context {
      color: #c9d1d9;
    }
"""


def build_page(entries: List[Entry], source_path: Path) -> str:
    cards_html = "\n".join(entry_to_html(entry) for entry in entries)
    total = len(entries)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Conversation Viewer</title>
  <style>
{BASE_CSS}
  </style>
</head>
<body>
  <header>
    <div class="header-top">
      <h1>Conversation Viewer</h1>
      <div class="header-actions">
        <a href="/run_code_log.html" class="nav-button">View run_code uploads</a>
        <button id="toggle-meta" class="meta-toggle" type="button">Hide meta blocks</button>
      </div>
    </div>
    <p>Source: {html.escape(str(source_path))} · {total} entries</p>
  </header>
  <div class="container">
    {cards_html}
  </div>
  <script>
    (() => {{
      const btn = document.getElementById("toggle-meta");
      if (!btn) return;
      let hidden = false;
      const update = () => {{
        document.body.classList.toggle("meta-hidden", hidden);
        btn.textContent = hidden ? "Show meta blocks" : "Hide meta blocks";
      }};
      btn.addEventListener("click", () => {{
        hidden = !hidden;
        update();
      }});
      update();
    }})();
  </script>
</body>
</html>
"""


def entry_to_html(entry: Entry) -> str:
    classes = " ".join([entry.css_class, *entry.extra_classes]).strip()
    return (
        f'<article class="entry {classes}">'
        f"<header>"
        f"<div>{html.escape(entry.label)}</div>"
        f"<small>{html.escape(entry.timestamp)} · line {entry.lineno} · {html.escape(entry.raw_type)}</small>"
        f"</header>"
        f"<div>{entry.body_html}</div>"
        f"</article>"
    )


def build_run_code_page(source_path: Path) -> str:
    uploads = extract_run_code_uploads(source_path)
    total = len(uploads)
    summary_section = render_upload_summary(uploads)
    diffs_section = ""
    if uploads:
        try:
            diffs = build_upload_git_history(uploads)
        except RuntimeError as exc:
            diffs_section = (
                "<section class='panel'>"
                "<h2>Commit diffs</h2>"
                f"<div class='error-banner'>Failed to build git history: {html.escape(str(exc))}</div>"
                "</section>"
            )
        else:
            diff_cards = "".join(
                f"<div class='diff-card'><h3>{html.escape(label)}</h3>{render_diff(diff)}</div>"
                for label, diff in diffs
            )
            diffs_section = f"<section class='panel'><h2>Commit diffs</h2>{diff_cards}</section>"
    else:
        diffs_section = ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>run_code uploads</title>
  <style>
{BASE_CSS}
  </style>
</head>
<body>
  <header>
    <div class="header-top">
      <h1>run_code uploads</h1>
      <div class="header-actions">
        <a href="/index.html" class="nav-button secondary">Back to entries</a>
      </div>
    </div>
    <p>Source: {html.escape(str(source_path))} · {total} uploads</p>
  </header>
  <div class="container">
    {summary_section}
    {diffs_section}
  </div>
</body>
</html>
"""


def render_upload_summary(uploads: List[RunCodeUpload]) -> str:
    if not uploads:
        return (
            "<section class='panel'>"
            "<h2>Captured uploads</h2>"
            "<p>No calls to mcp__kernelmcp__vm_compile_c_and_upload were found in this log.</p>"
            "</section>"
        )
    rows = []
    for upload in uploads:
        code_details = (
            f"<details><summary>{len(upload.code)} chars</summary>"
            f"<pre>{html.escape(upload.code)}</pre></details>"
            if upload.code
            else "<em>empty</em>"
        )
        flags_details = (
            f"<details><summary>{len(upload.flags)} chars</summary>"
            f"<pre>{html.escape(upload.flags)}</pre></details>"
            if upload.flags
            else "<em>empty</em>"
        )
        rows.append(
            "<tr>"
            f"<td>{upload.index}</td>"
            f"<td>{html.escape(upload.timestamp)}</td>"
            f"<td>line {upload.lineno}</td>"
            f"<td>{code_details}</td>"
            f"<td>{flags_details}</td>"
            "</tr>"
        )
    table = (
        "<table class='uploads-table'>"
        "<thead><tr><th>#</th><th>Timestamp</th><th>Location</th><th>Code</th><th>Flags</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )
    return f"<section class='panel'><h2>Captured uploads ({len(uploads)})</h2>{table}</section>"


def extract_run_code_uploads(source_path: Path) -> List[RunCodeUpload]:
    uploads: List[RunCodeUpload] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "response_item":
                continue
            payload = record.get("payload") or {}
            if payload.get("type") != "function_call":
                continue
            if payload.get("name") != TARGET_RUN_CODE_FN:
                continue
            args_raw = payload.get("arguments")
            args = try_parse_json(args_raw)
            if not isinstance(args, dict):
                continue
            code_value = args.get("code", "")
            code_str = str("" if code_value is None else code_value)
            flags_value = args.get("flags", "")
            if isinstance(flags_value, (list, tuple)):
                flags_str = "\n".join(str(item) for item in flags_value)
            else:
                flags_str = "" if flags_value is None else str(flags_value)
            uploads.append(
                RunCodeUpload(
                    index=len(uploads) + 1,
                    timestamp=record.get("timestamp", "unknown"),
                    lineno=lineno,
                    code=code_str,
                    flags=str(flags_str),
                )
            )
    return uploads


def build_upload_git_history(uploads: List[RunCodeUpload]) -> List[tuple[str, str]]:
    if not uploads:
        return []
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir)
        git_env = build_git_env()
        run_git_command(["init", "-q"], repo, git_env)
        code_path = repo / "code.c"
        flags_path = repo / "flags.txt"
        for upload in uploads:
            code_path.write_text(upload.code, encoding="utf-8")
            flags_path.write_text(upload.flags, encoding="utf-8")
            run_git_command(["add", "code.c", "flags.txt"], repo, git_env)
            commit_message = f"upload {upload.index}"
            run_git_command(
                ["commit", "-m", commit_message, "--allow-empty"],
                repo,
                git_env,
            )
        revs_output = run_git_command(["rev-list", "--reverse", "HEAD"], repo, git_env)
        revs = [rev for rev in revs_output.strip().splitlines() if rev]
        diffs: List[tuple[str, str]] = []
        for rev, upload in zip(revs, uploads):
            short = run_git_command(["rev-parse", "--short", rev], repo, git_env).strip()
            diff = run_git_command(["show", "--stat", "--patch", rev], repo, git_env)
            label = f"{short} · upload {upload.index}"
            diffs.append((label, diff))
        return diffs


def build_git_env() -> dict:
    env = os.environ.copy()
    env.setdefault("GIT_AUTHOR_NAME", "RunCodeLogger")
    env.setdefault("GIT_AUTHOR_EMAIL", "run-code@example.com")
    env.setdefault("GIT_COMMITTER_NAME", env["GIT_AUTHOR_NAME"])
    env.setdefault("GIT_COMMITTER_EMAIL", env["GIT_AUTHOR_EMAIL"])
    return env


def run_git_command(args: List[str], cwd: Path, env: dict) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "git command failed"
        raise RuntimeError(f"git {' '.join(args)}: {stderr}")
    return result.stdout


def start_server(
    port: int,
    index_builder: Callable[[], str],
    run_code_builder: Callable[[], str],
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                body_builder = index_builder
            elif self.path == "/run_code_log.html":
                body_builder = run_code_builder
            else:
                self.send_error(404, "Not Found")
                return
            body = body_builder().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            # Keep stdout clean; use stderr for concise logs.
            sys.stderr.write("Server: " + format % args + "\n")

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Serving log on http://127.0.0.1:{port}")
    server.serve_forever()


def main() -> None:
    args = parse_args()
    source_path = Path(args.log).expanduser().resolve()
    if not source_path.exists():
        print(f"Log file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    entries = load_entries(source_path)
    if not entries:
        print("No entries were parsed from the log.", file=sys.stderr)
        sys.exit(1)

    def page_builder() -> str:
        return build_page(entries, source_path)

    def run_code_builder() -> str:
        return build_run_code_page(source_path)

    start_server(args.port, page_builder, run_code_builder)


if __name__ == "__main__":
    main()
