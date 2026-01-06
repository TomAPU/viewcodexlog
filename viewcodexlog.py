#!/usr/bin/env python3
"""
Simple viewer for Codex CLI JSONL logs.

Usage:
    python3 viewcodexlog.py -p <port>
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import parse_qs, quote, urlparse


@dataclass
class Entry:
    timestamp: str
    label: str
    body_html: str
    css_class: str
    raw_type: str
    lineno: int
    extra_classes: List[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Codex JSONL logs as HTML.")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server (default: 8000).",
    )
    parser.add_argument(
        "-l",
        "--log",
        help=(
            "Optional base directory. If set, collect JSONL under <dir>/*/sessions/YYYY/MM/DD/*.jsonl; "
            "otherwise use current directory (non-recursive)."
        ),
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
            if (
                isinstance(parsed_output, dict)
                and isinstance(parsed_output.get("result"), list)
                and len(parsed_output["result"]) > 10
            ):
                rendered = render_structured_data(parsed_output)
                output_html = f'<div class="result-scroll">{rendered}</div>'
            else:
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


def build_page(entries: List[Entry], source_path: Path) -> str:
    cards_html = "\n".join(entry_to_html(entry) for entry in entries)
    total = len(entries)
    featured = entry_to_html(entries[-1]) if entries else ""
    featured_block = (
        f'<section class="featured-entry">{featured}<div class="divider-thin"></div></section>'
        if featured
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Conversation Viewer</title>
    <link rel="stylesheet" href="/style.css">
</head>
<body>
    <header>
        <div class="header-top">
            <h1><a class="logo-link" href="/index.html">Conversation Viewer</a></h1>
            <div class="header-actions">
                <a class="home-button" href="/index.html">Back to Home</a>
            </div>
        </div>
        <p>Source: {html.escape(str(source_path))} · {total} entries</p>
    </header>
    <div class="container">
        {featured_block}
        {cards_html}
    </div>
    <button id="back-to-top" class="back-to-top" aria-label="Back to top">↑</button>
    <script>
      (() => {{
        const btn = document.getElementById('back-to-top');
        if (!btn) return;
        const toggle = () => {{
          btn.classList.toggle('visible', window.scrollY > 240);
        }};
        window.addEventListener('scroll', toggle);
        btn.addEventListener('click', () => window.scrollTo({{ top: 0, behavior: 'smooth' }}));
        toggle();
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

def scan_jsonl_files(base_dir: Path) -> List[Path]:
    files = []
    for item in base_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".jsonl":
            files.append(item)
    return sorted(files)


def scan_sessions_jsonl(base_dir: Path) -> List[Path]:
    # Pattern: <base>/<case>/sessions/YYYY/MM/DD/*.jsonl
    pattern = "*/sessions/*/*/*/*.jsonl"
    return sorted(p for p in base_dir.glob(pattern) if p.is_file())


def format_file_info(path: Path) -> tuple[str, str]:
    try:
        stat = path.stat()
    except OSError:
        return ("n/a", "n/a")
    size_kb = stat.st_size / 1024
    size_label = f"{size_kb:.1f} KB" if size_kb >= 0.1 else f"{stat.st_size} B"
    mtime_label = datetime.fromtimestamp(stat.st_mtime).isoformat(" ", "seconds")
    return (size_label, mtime_label)


def build_index_page(base_dir: Path, files: List[Path], default_log: Path) -> str:
    rows = []
    for path in files:
        size_label, mtime_ts = format_file_info(path)
        name = path.name
        rel = path.relative_to(base_dir).as_posix()
        view_href = f"/view.html?file={quote(rel)}"
        default_badge = "" if path != default_log else "<span class=\"status-chip status-pending\">default</span>"
        rows.append(
            "<tr>"
            f"<td><a href=\"{view_href}\">{html.escape(rel)}</a></td>"
            f"<td>{size_label}</td>"
            f"<td>{mtime_ts}</td>"
            f"<td>{default_badge}</td>"
            "</tr>"
        )

    table = (
        "<table class=\"uploads-table\">"
        "<thead><tr><th>File</th><th>Size</th><th>Modified</th><th></th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    ) if rows else "<p>No .jsonl files found in this directory.</p>"

    return f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset="utf-8">
    <title>Log index</title>
    <link rel="stylesheet" href="/style.css">
</head>
<body>
    <header>
        <div class=\"header-top\">
            <h1><a class=\"logo-link\" href=\"/index.html\">Log index</a></h1>
            <div class=\"header-actions\">
                <a class=\"home-button\" href=\"/index.html\">Back to Home</a>
            </div>
        </div>
        <p>Root: {html.escape(str(base_dir))}</p>
    </header>
    <div class=\"container\">
        <section class=\"panel\">
            <h2>Available JSONL logs</h2>
            {table}
        </section>
    </div>
    <button id=\"back-to-top\" class=\"back-to-top\" aria-label=\"Back to top\">↑</button>
    <script>
      (() => {{
        const btn = document.getElementById('back-to-top');
        if (!btn) return;
        const toggle = () => {{
          btn.classList.toggle('visible', window.scrollY > 240);
        }};
        window.addEventListener('scroll', toggle);
        btn.addEventListener('click', () => window.scrollTo({{ top: 0, behavior: 'smooth' }}));
        toggle();
      }})();
    </script>
</body>
</html>
"""


def resolve_log_path(query: dict, base_dir: Path, default_log: Path) -> Optional[Path]:
    file_values = query.get("file")
    candidate = file_values[0] if file_values else default_log.relative_to(base_dir).as_posix()
    target = (base_dir / candidate).resolve()
    base = base_dir.resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return None
    if not target.is_file():
        return None
    return target


def start_server(
    port: int,
    base_dir: Path,
    default_log: Path,
    style_path: Path,
    nested_sessions: bool,
) -> None:
    try:
        css_bytes = style_path.read_bytes()
    except OSError:
        css_bytes = b""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path == "/style.css":
                if not css_bytes:
                    self.send_error(404, "CSS not found")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/css; charset=utf-8")
                self.send_header("Content-Length", str(len(css_bytes)))
                self.end_headers()
                self.wfile.write(css_bytes)
                return

            files = scan_sessions_jsonl(base_dir) if nested_sessions else scan_jsonl_files(base_dir)
            if not files:
                self.send_error(404, "No JSONL files found")
                return
            current_default = default_log if default_log.exists() else files[0]

            if path in {"/", "/index.html"}:
                body = build_index_page(base_dir, files, current_default)
            elif path in {"/view", "/view.html"}:
                log_path = resolve_log_path(query, base_dir, current_default)
                if not log_path:
                    self.send_error(404, "Log file not found")
                    return
                entries = load_entries(log_path)
                body = build_page(entries, log_path)
            else:
                self.send_error(404, "Not Found")
                return

            body_bytes = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        def log_message(self, format: str, *args: object) -> None:
            # Keep stdout clean; use stderr for concise logs.
            sys.stderr.write("Server: " + format % args + "\n")

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Serving log on http://127.0.0.1:{port}")
    server.serve_forever()


def main() -> None:
    args = parse_args()
    nested = bool(args.log)
    base_dir = Path(args.log).expanduser().resolve() if args.log else Path.cwd().resolve()
    candidates = scan_sessions_jsonl(base_dir) if nested else scan_jsonl_files(base_dir)
    if not candidates:
        msg = (
            "No .jsonl files found in current directory." if not nested
            else "No .jsonl files found under */sessions/YYYY/MM/DD/."
        )
        print(msg, file=sys.stderr)
        sys.exit(1)
    default_log = candidates[0]

    # Parse once to surface malformed logs early, but keep serving the index.
    try:
        _ = load_entries(default_log)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to parse default log: {exc}", file=sys.stderr)

    style_path = Path(__file__).with_name("style.css")
    start_server(args.port, base_dir, default_log, style_path, nested)


if __name__ == "__main__":
    main()
