#!/usr/bin/env python3
"""
Simple viewer for Codex CLI JSONL logs.

Usage:
    python3 viewcodexlog.py -l <logfile.jsonl> -p <port>
    python3 viewcodexlog.py -l <logfile.jsonl> -o <output.html>
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Iterable, List, Optional
from urllib.parse import parse_qs, quote, urlsplit


@dataclass
class Entry:
    timestamp: str
    label: str
    body_html: str
    css_class: str
    raw_type: str
    lineno: int
    extra_classes: List[str] = field(default_factory=list)
    anchor_id: Optional[str] = None
    tool_name: Optional[str] = None
    next_tool_anchor: Optional[str] = None
    call_id: Optional[str] = None


@dataclass
class RunCodeUpload:
    index: int
    timestamp: str
    lineno: int
    code: str
    flags: str


@dataclass
class SessionLog:
    session_id: str
    path: Path
    rel_path: str
    timestamp: str
    time_label: str
    title: str
    summary: str


TARGET_RUN_CODE_FN = "mcp__kernelmcp__vm_compile_c_and_upload"
DEFAULT_SESSIONS_DIR = Path.home() / ".codex" / "sessions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Codex JSONL logs as HTML.")
    parser.add_argument(
        "-l",
        "--log",
        help="Path to a single JSONL conversation log (optional).",
    )
    parser.add_argument(
        "--sessions-dir",
        default=str(DEFAULT_SESSIONS_DIR),
        help="Directory to scan for JSONL session logs when --log is not set.",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server (default: 8000).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write HTML output to this file or directory instead of serving.",
    )
    return parser.parse_args()


def parse_iso_timestamp_safe(value: object) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return parse_iso_timestamp(value)
    except ValueError:
        return None


def session_time_from_filename(path: Path) -> Optional[str]:
    match = re.search(r"rollout-(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})-", path.name)
    if not match:
        return None
    raw = match.group(1)
    date_part, time_part = raw.split("T", 1)
    return f"{date_part}T{time_part.replace('-', ':')}Z"


def format_session_time_label(timestamp: str) -> str:
    return format_timestamp_for_display(timestamp)


def format_timestamp_for_display(timestamp: str) -> str:
    dt = parse_iso_timestamp_safe(timestamp)
    if dt is None:
        return timestamp or "unknown time"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local_dt = dt.astimezone()
    tz_label = local_dt.tzname() or local_dt.strftime("%z")
    return local_dt.strftime("%Y-%m-%d %H:%M:%S ") + tz_label


def compact_text(text: object, limit: int = 96) -> str:
    if text is None:
        return ""
    normalized = " ".join(str(text).strip().split())
    if not normalized:
        return ""
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "..."


def is_session_bootstrap_text(text: str) -> bool:
    lowered = text.lower()
    noisy_markers = [
        "agents.md instructions",
        "<environment_context>",
        "<collaboration_mode>",
        "<permissions instructions>",
        "## skills",
        "how to use skills",
    ]
    return any(marker in lowered for marker in noisy_markers)


def extract_first_user_prompt(payload: dict) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    texts = extract_text_chunks(content)
    if not texts:
        return ""
    return compact_text(texts[0], limit=120)


def build_session_log(path: Path, root_dir: Path) -> Optional[SessionLog]:
    timestamp = ""
    title = ""
    summary = ""
    fallback_prompt = ""
    try:
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not timestamp:
                    rec_ts = record.get("timestamp")
                    if isinstance(rec_ts, str):
                        timestamp = rec_ts
                rectype = record.get("type")
                payload = record.get("payload") or {}
                if not isinstance(payload, dict):
                    payload = {}
                if rectype == "session_meta":
                    payload_ts = payload.get("timestamp")
                    if isinstance(payload_ts, str):
                        timestamp = payload_ts if not timestamp else timestamp
                    candidate_title = (
                        payload.get("title")
                        or payload.get("name")
                        or payload.get("topic")
                    )
                    if isinstance(candidate_title, str) and candidate_title.strip():
                        title = compact_text(candidate_title, limit=120)
                    candidate_summary = payload.get("summary")
                    if isinstance(candidate_summary, str) and candidate_summary.strip():
                        summary = compact_text(candidate_summary, limit=120)
                elif rectype == "response_item":
                    if payload.get("type") == "message" and payload.get("role") == "user":
                        prompt = extract_first_user_prompt(payload)
                        if prompt:
                            if not fallback_prompt:
                                fallback_prompt = prompt
                            if not is_session_bootstrap_text(prompt):
                                summary = summary or prompt
                                title = title or prompt
                                break
                elif rectype == "event_msg":
                    if payload.get("type") == "user_message":
                        message = compact_text(payload.get("message"), limit=120)
                        if message:
                            if not fallback_prompt:
                                fallback_prompt = message
                            if not is_session_bootstrap_text(message):
                                summary = summary or message
                                title = title or message
                                break
                if lineno >= 1200 and timestamp:
                    break
    except OSError:
        return None

    if not timestamp:
        timestamp = session_time_from_filename(path) or ""
    if not summary and fallback_prompt:
        summary = fallback_prompt
    if not title:
        title = summary or path.stem
    if not summary:
        summary = title
    try:
        rel_path = str(path.relative_to(root_dir))
    except ValueError:
        rel_path = str(path)
    session_id = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    return SessionLog(
        session_id=session_id,
        path=path,
        rel_path=rel_path,
        timestamp=timestamp,
        time_label=format_session_time_label(timestamp),
        title=title,
        summary=summary,
    )


def discover_session_logs(root_dir: Path) -> List[SessionLog]:
    if not root_dir.exists():
        return []
    sessions: List[SessionLog] = []
    for path in sorted(root_dir.rglob("*.jsonl")):
        if not path.is_file():
            continue
        session = build_session_log(path.resolve(), root_dir)
        if session:
            sessions.append(session)
    sessions.sort(key=session_sort_key, reverse=True)
    return sessions


def session_sort_key(session: SessionLog) -> datetime:
    dt = parse_iso_timestamp_safe(session.timestamp)
    if dt is not None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    try:
        return datetime.fromtimestamp(session.path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def resolve_session_or_default(
    sessions: List[SessionLog], session_id: Optional[str]
) -> Optional[SessionLog]:
    if not sessions:
        return None
    if session_id:
        for session in sessions:
            if session.session_id == session_id:
                return session
    return sessions[0]


def load_entries(path: Path) -> List[Entry]:
    entries: List[Entry] = []
    pending_calls: dict[str, Entry] = {}
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
            if not entry:
                continue

            if entry.raw_type == "response_item/function_call" and entry.call_id:
                pending_calls[entry.call_id] = entry
            elif (
                entry.raw_type == "response_item/function_call_output"
                and entry.call_id
                and entry.call_id in pending_calls
            ):
                call_entry = pending_calls.pop(entry.call_id)
                # Merge output into call
                call_entry.label = f"Call & Output · {call_entry.tool_name or 'unknown'}"
                call_entry.body_html = (
                    f'<div class="call-section">{call_entry.body_html}</div>'
                    f'<div class="output-divider"></div>'
                    f'<div class="output-section">{entry.body_html}</div>'
                )
                entries.append(call_entry)
            else:
                entries.append(entry)

    # Any leftover pending calls that never got output
    for entry in pending_calls.values():
        entries.append(entry)

    # Re-sort entries by lineno because pending_calls might have deferred them
    entries.sort(key=lambda e: e.lineno)
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
            extra_classes=["collapsible-meta", "turn-context-block"],
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
            anchor_id=f"entry-{lineno}",
            tool_name=name,
            call_id=call_id,
        )

    if subtype == "function_call_output":
        call_id = payload.get("call_id", "n/a")
        output = payload.get("output")
        parsed_output = try_parse_json(output)
        if parsed_output is not None:
            output_html = render_structured_data(parsed_output)
        elif output is None:
            output_html = "<em>no output</em>"
        elif isinstance(output, str):
            output_html = render_maybe_truncated_text(output)
        else:
            output_html = render_scalar(output)
        body = f"{output_html}"
        return Entry(
            timestamp=timestamp,
            label="Function output",
            body_html=body,
            css_class="entry-tool",
            raw_type="response_item/function_call_output",
            lineno=lineno,
            call_id=call_id,
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
            extra_classes=["collapsible-meta", "summary-block"],
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
            extra_classes=["event-block"],
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
            extra_classes=["collapsible-meta", "token-usage-block"],
        )

    return Entry(
        timestamp=timestamp,
        label=f"Event ({subtype or 'unknown'})",
        body_html=format_payload(payload),
        css_class="entry-system",
        raw_type=f"event_msg/{subtype or 'unknown'}",
        lineno=lineno,
        extra_classes=["event-block"],
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


def parse_iso_timestamp(value: str) -> datetime:
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def choose_tick_interval(duration_minutes: float) -> int:
    candidates = [1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300]
    for interval in candidates:
        if duration_minutes / interval <= 8:
            return interval
    return candidates[-1]


def color_for_tool(tool_name: str) -> str:
    digest = hashlib.sha1(tool_name.encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) % 360
    return f"hsl({hue}, 70%, 55%)"


def build_tool_color_map(tool_names: List[str]) -> dict:
    seen = []
    colors = {}
    base_hue = 210
    golden = 137.508
    for name in sorted(set(tool_names)):
        idx = len(seen)
        hue = (base_hue + idx * golden) % 360
        colors[name] = f"hsl({int(round(hue))}, 72%, 48%)"
        seen.append(name)
    return colors


def format_duration_minutes(minutes: float) -> str:
    if minutes < 1:
        return f"{int(round(minutes * 60))}s"
    hours = int(minutes // 60)
    rem = int(round(minutes - hours * 60))
    if rem == 60:
        hours += 1
        rem = 0
    if hours and rem:
        return f"{hours}h {rem}m"
    if hours:
        return f"{hours}h"
    return f"{rem}m"


def render_structured_data(data: object) -> str:
    if isinstance(data, dict):
        if data.get("type") == "code":
            return render_code_block(data)
        items = []
        for key, value in data.items():
            key_str = str(key)
            val_html = render_structured_data(value)
            row_class = f"kv-row-{html.escape(key_str.lower().replace('_', '-'))}"
            items.append(
                f'<div class="kv-row {row_class}">'
                f'<span class="kv-key">{html.escape(key_str)}</span>'
                f'<div class="kv-val">{val_html}</div>'
                f'</div>'
            )
        return f'<div class="kv-list">{"".join(items)}</div>'
    if isinstance(data, list):
        items = "".join(
            f"<li>{render_structured_data(item)}</li>" for item in data)
        return f'<ul class="list-nested">{items}</ul>'
    return render_scalar(data)


def render_maybe_truncated_text(text: str, limit: int = 20, show: int = 10) -> str:
    if not isinstance(text, str):
        text = str(text)
    lines = text.splitlines()
    if len(lines) <= limit:
        return format_pre(text)
    
    visible = "\n".join(lines[:show])
    hidden = "\n".join(lines[show:])
    
    return (
        f'<div class="truncated-text">'
        f'{format_pre(visible)}'
        f'<details class="truncated-details">'
        f'<summary class="text-secondary" style="cursor:pointer; font-size:0.85rem; padding:0.5rem 0;">'
        f'↓ Show {len(lines) - show} more lines'
        f'</summary>'
        f'{format_pre(hidden)}'
        f'</details>'
        f'</div>'
    )


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
    copy_btn = '<button type="button" class="copy-btn" onclick="copyCode(this)">Copy</button>'
    return f'<div class="code-block">{header}{copy_btn}<pre><code>{html.escape(code)}</code></pre></div>'


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


def render_tool_timeline(entries: List[Entry]) -> str:
    tool_events: List[tuple[Entry, datetime]] = []
    for entry in entries:
        if entry.tool_name and entry.anchor_id:
            tool_events.append((entry, parse_iso_timestamp(entry.timestamp)))
    if not tool_events:
        return ""
    color_map = build_tool_color_map(
        [entry.tool_name for entry, _ in tool_events if entry.tool_name]
    )
    tool_events.sort(key=lambda item: item[1])
    start_ts = tool_events[0][1]
    end_ts = tool_events[-1][1]
    duration_minutes = max((end_ts - start_ts).total_seconds() / 60.0, 0.01)
    tick_interval = choose_tick_interval(duration_minutes)
    events_payload = []
    for entry, when in tool_events:
        minutes = (when - start_ts).total_seconds() / 60.0
        color = color_for_tool(entry.tool_name)
        events_payload.append(
            {
                "id": entry.anchor_id,
                "tool": entry.tool_name,
                "timestamp": entry.timestamp,
                "timestampLabel": format_timestamp_for_display(entry.timestamp),
                "minutes": round(minutes, 3),
                "color": color_map.get(entry.tool_name, color),
            }
        )
    config = {
        "events": events_payload,
        "totalMinutes": duration_minutes,
        "tickInterval": tick_interval,
    }
    config_json = json.dumps(config).replace("</", "<\\/")
    start_label = html.escape(format_timestamp_for_display(tool_events[0][0].timestamp))
    end_label = html.escape(format_timestamp_for_display(tool_events[-1][0].timestamp))
    duration_label = html.escape(format_duration_minutes(duration_minutes))
    return f"""
  <section class="timeline-panel" id="tool-timeline">
    <div class="timeline-inner">
      <div class="timeline-header">
        <div>
          <strong>Tool timeline</strong>
          <div class="timeline-range">{start_label} → {end_label} · {duration_label}</div>
        </div>
        <button type="button" class="nav-button secondary" id="timeline-hide-btn">Hide</button>
      </div>
      <div class="timeline-scroll" id="timeline-scroll">
        <div class="timeline-track" id="timeline-track"></div>
        <div class="timeline-axis" id="timeline-axis"></div>
      </div>
      <div class="legend-actions">
        <button type="button" class="nav-button secondary small" id="legend-select-all">Select all</button>
        <button type="button" class="nav-button secondary small" id="legend-clear-all">Unselect all</button>
      </div>
      <div class="timeline-legend" id="timeline-legend"></div>
    </div>
  </section>
  <button class="timeline-toggle-btn hidden" id="timeline-toggle-btn" type="button">Show tool timeline</button>
  <script>
    const setupTimeline = () => {{
      const cfg = {config_json};
      const panel = document.getElementById("tool-timeline");
      const showBtn = document.getElementById("timeline-toggle-btn");
      const hideBtn = document.getElementById("timeline-hide-btn");
      const scrollBox = document.getElementById("timeline-scroll");
      const track = document.getElementById("timeline-track");
      const axis = document.getElementById("timeline-axis");
      const legend = document.getElementById("timeline-legend");
      const selectAllBtn = document.getElementById("legend-select-all");
      const clearAllBtn = document.getElementById("legend-clear-all");
      if (!panel || !track || !axis || !legend || !scrollBox || !cfg?.events?.length) return;

      const total = Math.max(cfg.totalMinutes, 0.01);
      const laneHeight = 18;
      const minGapPx = 18;
      const pxPerMinute = 28;
      const minWidth = 800;
      let totalWidth = minWidth;
      const bucketSize = 18; // px grouping for near-simultaneous calls
      const maxStackPerColumn = 5;
      const columnOffset = 12; // px horizontal nudge when stacking over 5

      const scrollEntryIntoView = (id) => {{
        const target = document.getElementById(id);
        if (!target) return null;
        const gap = panel.classList.contains("hidden") ? 8 : (panel.offsetHeight + 8);
        const top = target.getBoundingClientRect().top + window.scrollY - gap;
        window.scrollTo({{ top, behavior: "smooth" }});
        return target;
      }};

      const highlightEntry = (id) => {{
        const target = scrollEntryIntoView(id);
        if (!target) return;
        target.classList.add("entry-highlight");
        setTimeout(() => target.classList.remove("entry-highlight"), 2200);
      }};

      const highlightMarker = (id) => {{
        const marker = track.querySelector(`.timeline-event[data-id=\"${{id}}\"]`);
        if (!marker) return;
        marker.classList.add("highlight");
        setTimeout(() => marker.classList.remove("highlight"), 1200);
      }};

      const applyFilters = () => {{
        const activeTools = new Set();
        legend.querySelectorAll('input[type="checkbox"]').forEach((cb) => {{
          if (cb.checked) activeTools.add(cb.value);
        }});
        track.querySelectorAll('.timeline-event').forEach((node) => {{
          const tool = node.dataset.tool;
          node.style.display = activeTools.has(tool) ? "block" : "none";
        }});
      }};

      const placeEvents = () => {{
        const viewport = scrollBox.clientWidth || window.innerWidth || 1;
        const baseWidth = Math.max(minWidth, viewport, total * pxPerMinute);
        const inset = 14;
        let usable = Math.max(0, baseWidth - inset * 2);
        totalWidth = baseWidth + inset * 2;
        track.innerHTML = "";
        legend.innerHTML = "";
        axis.innerHTML = "";

        const buckets = new Map();
        const seenTools = new Map();
        let maxX = 0;
        cfg.events.forEach((ev) => {{
          const rawX = inset + Math.max(0, Math.min(usable, (ev.minutes / total) * usable));
          const bucket = Math.floor(rawX / bucketSize);
          const count = buckets.get(bucket) || 0;
          const columnIdx = Math.floor(count / maxStackPerColumn);
          const laneIdx = count % maxStackPerColumn;
          const x = rawX + columnIdx * columnOffset;
          buckets.set(bucket, count + 1);
          maxX = Math.max(maxX, x + 16);

          const node = document.createElement("button");
          node.type = "button";
          node.className = "timeline-event";
          node.title = `${{ev.tool}} · ${{ev.timestampLabel || ev.timestamp}}`;
          node.style.left = `${{x}}px`;
          node.style.top = `${{10 + laneIdx * laneHeight}}px`;
          node.style.backgroundColor = ev.color;
          node.dataset.tool = ev.tool;
          node.addEventListener("click", () => {{
            highlightEntry(ev.id);
            highlightMarker(ev.id);
          }});
          node.dataset.id = ev.id;
          track.appendChild(node);
          seenTools.set(ev.tool, ev.color);
        }});

        totalWidth = Math.max(totalWidth, maxX + inset + 10);
        track.style.width = `${{totalWidth}}px`;
        axis.style.width = `${{totalWidth}}px`;

        const maxCount = buckets.size ? Math.max(...Array.from(buckets.values())) : 0;
        const height = Math.max(50, Math.min(maxStackPerColumn, maxCount || 1) * laneHeight + 24);
        track.style.height = `${{height}}px`;

        const usableTicks = Math.max(0, totalWidth - inset * 2);
        const interval = cfg.tickInterval || total;
        for (let m = 0; m <= cfg.totalMinutes + interval * 0.25; m += interval) {{
          const tick = document.createElement("div");
          tick.className = "timeline-tick";
          const px = inset + Math.min(usableTicks, Math.max(0, (m / total) * usableTicks));
          tick.style.left = `${{px}}px`;
          if (m === 0) {{
            tick.classList.add("tick-start");
          }}
          tick.textContent = formatTick(m);
          axis.appendChild(tick);
        }}

        for (const [tool, color] of seenTools.entries()) {{
          const item = document.createElement("label");
          item.className = "legend-item";
          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.value = tool;
          cb.checked = true;
          const swatch = document.createElement("span");
          swatch.className = "swatch";
          swatch.style.backgroundColor = color;
          const label = document.createElement("span");
          label.textContent = tool;
          item.appendChild(cb);
          item.appendChild(swatch);
          item.appendChild(label);
          legend.appendChild(item);
          cb.addEventListener("change", applyFilters);
        }}
        applyFilters();
      }};

      const formatTick = (minutes) => {{
        if (minutes >= 60) {{
          const h = Math.floor(minutes / 60);
          const m = Math.round(minutes - h * 60);
          return m ? `${{h}}h ${{m}}m` : `${{h}}h`;
        }}
        if (minutes < 1) return `${{Math.round(minutes * 60)}}s`;
        return `${{Math.round(minutes)}}m`;
      }};

      const updateLayout = () => {{
        placeEvents();
        const maxScroll = Math.max(0, totalWidth - scrollBox.clientWidth);
        scrollBox.scrollLeft = Math.min(scrollBox.scrollLeft, maxScroll);
      }};

      const hidePanel = () => {{
        panel.classList.add("hidden");
        showBtn?.classList.remove("hidden");
      }};
      const showPanel = () => {{
        panel.classList.remove("hidden");
        showBtn?.classList.add("hidden");
        updateLayout();
      }};
      hideBtn?.addEventListener("click", hidePanel);
      showBtn?.addEventListener("click", showPanel);
      selectAllBtn?.addEventListener("click", () => {{
        legend.querySelectorAll('input[type="checkbox"]').forEach((cb) => cb.checked = true);
        applyFilters();
      }});
      clearAllBtn?.addEventListener("click", () => {{
        legend.querySelectorAll('input[type="checkbox"]').forEach((cb) => cb.checked = false);
        applyFilters();
      }});

      // Horizontal wheel scrolling
      scrollBox.addEventListener("wheel", (e) => {{
        if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {{
          scrollBox.scrollLeft += e.deltaY;
          e.preventDefault();
        }}
      }}, {{ passive: false }});

      const centerMarker = (id) => {{
        const marker = track.querySelector(`.timeline-event[data-id=\"${{id}}\"]`);
        if (!marker) return;
        const center = marker.offsetLeft - scrollBox.clientWidth / 2 + marker.offsetWidth / 2;
        scrollBox.scrollTo({{ left: center, behavior: 'smooth' }});
        highlightMarker(id);
      }};

      // Jump from entry to timeline marker or next invocation
      const wireEntryButtons = () => {{
        document.querySelectorAll('.jump-to-timeline').forEach((btn) => {{
          const targetId = btn.dataset.target;
          btn.addEventListener('click', () => {{
            centerMarker(targetId);
          }});
        }});

        document.querySelectorAll('.jump-to-next').forEach((btn) => {{
          const targetId = btn.dataset.target;
          btn.addEventListener('click', () => {{
            const nextEntry = scrollEntryIntoView(targetId);
            if (nextEntry) {{
              nextEntry.classList.add('entry-highlight');
              setTimeout(() => nextEntry.classList.remove('entry-highlight'), 2200);
            }}
            centerMarker(targetId);
          }});
        }});
      }};

      // Update sticky position based on header height
      const header = document.querySelector("header");
      const updateStickyPos = () => {{
        if (header && panel) {{
          panel.style.top = `${{header.offsetHeight}}px`;
        }}
      }};
      window.addEventListener("resize", updateStickyPos);
      updateStickyPos();

      updateLayout();
      wireEntryButtons();
      window.addEventListener("resize", updateLayout);
    }};
    if (document.readyState === "loading") {{
      document.addEventListener("DOMContentLoaded", setupTimeline);
    }} else {{
      setupTimeline();
    }}
  </script>
  """


def render_session_selector(
    sessions: List[SessionLog],
    active_session: SessionLog,
    target_path: str,
) -> str:
    if not sessions:
        return ""
    ordered_sessions = sorted(sessions, key=session_sort_key, reverse=True)
    items = []
    for session in ordered_sessions:
        active_class = " is-active" if session.session_id == active_session.session_id else ""
        href = f"{target_path}?sid={quote(session.session_id, safe='')}"
        title = compact_text(f"{session.time_label} | {session.title}", limit=180)
        meta = compact_text(f"{session.summary} · {session.rel_path}", limit=280)
        search_blob = " ".join(
            [
                session.time_label,
                session.title,
                session.summary,
                session.rel_path,
            ]
        ).lower()
        items.append(
            f'<li class="session-item{active_class}" data-search="{html.escape(search_blob)}">'
            f'<a href="{html.escape(href)}" class="session-link">'
            f'<div class="session-title">{html.escape(title)}</div>'
            f'<div class="session-meta">{html.escape(meta)}</div>'
            f"</a>"
            f"</li>"
        )
    rel_label = html.escape(active_session.rel_path)
    return f"""
    <div class="session-switcher">
      <label for="session-search">Search sessions</label>
      <input id="session-search" class="session-search-input" type="search" placeholder="Search by title, summary, path">
      <small id="session-search-status" class="session-path"></small>
      <ul id="session-list" class="session-list">
        {"\n".join(items)}
      </ul>
      <small class="session-path">Current: {rel_label}</small>
    </div>
    <script>
      (() => {{
        const input = document.getElementById("session-search");
        const list = document.getElementById("session-list");
        const status = document.getElementById("session-search-status");
        if (!input || !list || !status) return;
        const items = Array.from(list.querySelectorAll(".session-item"));
        const applyFilter = () => {{
          const query = input.value.trim().toLowerCase();
          let visible = 0;
          items.forEach((item) => {{
            const text = item.dataset.search || "";
            const hit = !query || text.includes(query);
            item.classList.toggle("hidden", !hit);
            if (hit) visible += 1;
          }});
          status.textContent = query
            ? `${{visible}} / ${{items.length}} sessions`
            : `${{items.length}} sessions`;
        }};
        input.addEventListener("input", applyFilter);
        input.addEventListener("keydown", (event) => {{
          if (event.key === "Escape") {{
            input.value = "";
            applyFilter();
          }}
        }});
        applyFilter();
      }})();
    </script>
    """


def third_user_message_index(entries: List[Entry]) -> Optional[int]:
    seen = 0
    for idx, entry in enumerate(entries):
        if entry.raw_type == "response_item/message" and entry.label == "Message · user":
            seen += 1
            if seen == 3:
                return idx
    return None


def prepare_entries_for_render(entries: List[Entry]) -> int:
    collapse_cutoff = third_user_message_index(entries)
    collapsed_count = 0
    if collapse_cutoff is not None and collapse_cutoff > 0:
        collapsed_count = collapse_cutoff
        for entry in entries[:collapse_cutoff]:
            if "pre-third-user" not in entry.extra_classes:
                entry.extra_classes.append("pre-third-user")

    next_for_tool: dict[str, Optional[str]] = {}
    for entry in reversed(entries):
        if entry.tool_name and entry.anchor_id:
            entry.next_tool_anchor = next_for_tool.get(entry.tool_name)
            next_for_tool[entry.tool_name] = entry.anchor_id
    return collapsed_count


BASE_CSS = """
    :root {
      /* Dark Theme (Default) */
      --bg-app: #0d1117;
      --bg-panel: #161b22;
      --bg-header: #161b22;
      --border-color: #30363d;
      --border-subtle: rgba(240, 246, 252, 0.1);
      --text-primary: #e6edf3;
      --text-secondary: #9198a1;
      --accent-color: #2f81f7;
      --accent-dim: rgba(47, 129, 247, 0.15);
      --success-color: #3fb950;
      --danger-color: #f85149;
      --warning-color: #d29922;
      --code-bg: #010409;
      --bg-hover: #21262d;
      --font-sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      --font-mono: "JetBrains Mono", "Fira Code", "SFMono-Regular", Consolas, monospace;
      --page-max-width: 1400px;
      --reading-width: 960px;
      --page-gutter: clamp(16px, 5vw, 40px);
      --radius-default: 8px;
      --shadow-sm: 0 1px 0 rgba(1, 4, 9, 0.04);
      --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    body.theme-light {
      /* Light Theme Override */
      --bg-app: #f6f8fa;
      --bg-panel: #ffffff;
      --bg-header: #ffffff;
      --border-color: #d0d7de;
      --border-subtle: rgba(31, 35, 40, 0.08);
      --text-primary: #1f2328;
      --text-secondary: #656d76;
      --accent-color: #0969da;
      --accent-dim: rgba(9, 105, 218, 0.1);
      --success-color: #1a7f37;
      --danger-color: #d1242f;
      --warning-color: #9a6700;
      --code-bg: #f6f8fa;
      --bg-hover: #f3f4f6;
      --shadow-sm: 0 1px 0 rgba(31, 35, 40, 0.04);
      --shadow-md: 0 8px 24px rgba(140, 149, 159, 0.2);
    }

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    *, *::before, *::after { box-sizing: border-box; }

    body {
      background-color: var(--bg-app);
      color: var(--text-primary);
      font-family: var(--font-sans);
      margin: 0;
      line-height: 1.65;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    header {
      position: sticky;
      top: 0;
      z-index: 200;
      background: var(--bg-header);
      border-bottom: 1px solid var(--border-color);
      padding: 1rem var(--page-gutter);
      backdrop-filter: blur(12px);
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    header > .header-top,
    header > .session-switcher,
    header > .controls-row {
      width: 100%;
      max-width: var(--page-max-width);
      margin-left: auto;
      margin-right: auto;
    }

    h1 {
      margin: 0;
      font-size: 1.5rem;
      font-weight: 600;
      letter-spacing: -0.02em;
      color: var(--text-primary);
    }

    .header-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      margin-bottom: 0.75rem;
    }

    .header-actions {
      display: flex;
      gap: 0.5rem;
    }

    .controls-row {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      align-items: center;
      margin-top: 0.5rem;
      padding-top: 0.5rem;
      border-top: 1px solid var(--border-subtle);
    }

    /* Inputs & Buttons */
    input[type="text"], input[type="search"] {
      background: var(--bg-panel);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      padding: 0.35rem 0.6rem;
      border-radius: var(--radius-default);
      font-size: 0.9rem;
      outline: none;
      transition: border-color 0.2s;
    }
    input[type="text"]:focus, input[type="search"]:focus {
      border-color: var(--accent-color);
      box-shadow: 0 0 0 2px var(--accent-dim);
    }

    button, .nav-button, .meta-toggle {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.35rem 0.85rem;
      border-radius: var(--radius-default);
      font-size: 0.85rem;
      font-weight: 500;
      cursor: pointer;
      text-decoration: none;
      transition: all 0.2s ease;
      border: 1px solid transparent;
      line-height: 1.2;
    }

    .nav-button {
      background: var(--accent-color);
      color: white;
      border-color: rgba(255,255,255,0.1);
    }
    .nav-button:hover {
      background: #388bfd;
      text-decoration: none;
    }
    
    .nav-button.secondary, .meta-toggle {
      background: var(--bg-panel);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
    }
    .nav-button.secondary:hover, .meta-toggle:hover {
      background: var(--bg-hover);
      border-color: var(--text-secondary);
    }

    .nav-button.small {
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
    }

    /* Session Switcher */
    .session-switcher {
        position: relative;
    }
    .session-switcher label {
        display: none; 
    }
    .session-search-input {
        width: 100%;
        max-width: 400px;
    }
    .session-list {
      position: absolute;
      top: 100%;
      left: 0;
      z-index: 300;
      background: var(--bg-panel);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-default);
      width: 100%;
      max-width: 600px;
      max-height: 400px;
      overflow-y: auto;
      box-shadow: var(--shadow-md);
      display: none;
      margin-top: 4px;
    }
    .session-search-input:focus + .session-path + .session-list,
    .session-list:hover {
        display: block;
    }
    
    .session-item {
        border-bottom: 1px solid var(--border-subtle);
    }
    .session-link {
        display: block;
        padding: 0.75rem;
        color: var(--text-primary);
        text-decoration: none;
    }
    .session-link:hover {
        background: var(--bg-hover);
    }
    .session-item.is-active .session-link {
        border-left: 3px solid var(--accent-color);
        background: var(--accent-dim);
    }
    .session-title { font-weight: 600; font-size: 0.9rem; margin-bottom: 0.2rem; }
    .session-link:hover {
        background: var(--bg-hover);
    }
    .session-item.is-active .session-link {
        border-left: 3px solid var(--accent-color);
        background: var(--accent-dim);
    }
    .session-title { font-weight: 600; font-size: 0.9rem; margin-bottom: 0.2rem; }
    .session-meta { font-size: 0.8rem; color: var(--text-secondary); }

    /* Timeline */
    .timeline-panel {
        background: var(--bg-panel);
        border-bottom: 1px solid var(--border-color);
        padding: 1rem var(--page-gutter);
        position: sticky;
        top: 0;
        z-index: 150;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: top 0.3s;
    }
    
    /* Ensure timeline is below header when header is visible */
    header ~ .timeline-panel {
        top: 0; /* Updated by JS if needed */
    }
        max-width: var(--page-max-width);
        margin: 0 auto;
    }
    .timeline-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1rem;
    }
    .timeline-range {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.2rem;
    }
    .timeline-scroll {
        overflow-x: auto;
        overflow-y: hidden;
        background: var(--bg-app);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-default);
        position: relative;
        margin-bottom: 1rem;
        cursor: grab;
    }
    .timeline-scroll:active { cursor: grabbing; }
    .timeline-track {
        position: relative;
        height: 80px; /* Adjusted by JS */
    }
    .timeline-axis {
        position: relative;
        height: 24px;
        border-top: 1px solid var(--border-subtle);
    }
    .timeline-event {
        position: absolute;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        border: 2px solid var(--bg-panel);
        cursor: pointer;
        padding: 0;
        z-index: 10;
        transition: transform 0.1s, box-shadow 0.1s;
    }
    .timeline-event:hover {
        transform: scale(1.4);
        z-index: 20;
        box-shadow: 0 0 0 4px var(--accent-dim);
    }
    .timeline-event.highlight {
        transform: scale(1.8);
        box-shadow: 0 0 0 6px var(--accent-color);
        z-index: 30;
    }
    .timeline-tick {
        position: absolute;
        bottom: 0;
        font-size: 0.7rem;
        color: var(--text-secondary);
        white-space: nowrap;
        transform: translateX(-50%);
        padding-bottom: 4px;
    }
    .timeline-tick::before {
        content: "";
        position: absolute;
        top: -24px;
        left: 50%;
        width: 1px;
        height: 24px;
        background: var(--border-subtle);
    }
    .timeline-tick.tick-start { transform: none; }
    .timeline-tick.tick-start::before { left: 0; }

    .legend-actions {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .timeline-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.8rem;
        color: var(--text-secondary);
        cursor: pointer;
        user-select: none;
    }
    .legend-item:hover { color: var(--text-primary); }
    .legend-item input { margin: 0; }
    .swatch {
        width: 10px;
        height: 10px;
        border-radius: 2px;
    }

    .timeline-toggle-btn {
        position: fixed;
        bottom: 1.5rem;
        right: 1.5rem;
        z-index: 1000;
        background: var(--accent-color);
        color: white;
        box-shadow: var(--shadow-md);
        border: none;
    }
    .timeline-toggle-btn:hover { background: #388bfd; }

    /* Entry highlighting */
    .entry-highlight {
        outline: 2px solid var(--accent-color);
        outline-offset: -2px;
        box-shadow: 0 0 20px var(--accent-dim);
    }

    /* Main Content */
    .container {
      padding: 1.5rem var(--page-gutter) 4rem;
      width: 100%;
      max-width: var(--page-max-width);
      margin: 0 auto;
    }

    /* Entry Cards */
    .entry {
      background: var(--bg-panel);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-default);
      margin-bottom: 1.5rem;
      width: min(100%, var(--reading-width));
      margin-left: auto;
      margin-right: auto;
      overflow: hidden;
      position: relative;
    }
    
    .entry header {
        background: var(--bg-panel);
        padding: 0.75rem 1.25rem;
        border-bottom: 1px solid var(--border-subtle);
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: static;
        box-shadow: none;
        backdrop-filter: none;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }
    
    .entry header strong {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    .entry-body {
        padding: 1.25rem 1.5rem;
        font-size: 1.05rem;
        line-height: 1.7;
        overflow-x: auto;
    }

    .output-divider {
        height: 1px;
        background: var(--border-subtle);
        margin: 1.25rem -1.5rem;
    }
    .call-section { margin-bottom: 0.75rem; }
    .output-section { margin-top: 0.75rem; }

    /* Colors by role */
    .entry.entry-user { border-left: 4px solid #1f6feb; }
    .entry.entry-assistant { border-left: 4px solid #8957e5; }
    .entry.entry-tool { border-left: 4px solid #d29922; }
    .entry.entry-system { border-left: 4px solid #8b949e; }
    
    /* KV List */
    .kv-list {
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
        margin: 0.75rem 0;
    }
    .kv-row {
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
    }
    .kv-key {
        font-weight: 600;
        color: var(--text-secondary);
        font-size: 0.9rem;
        width: 160px;
        flex-shrink: 0;
        padding-top: 0.1rem;
    }
    .kv-key::after { content: ":"; }
    .kv-val {
        flex: 1;
        font-size: 0.95rem;
        min-width: 0;
    }
    .kv-row-cmd .kv-val,
    .kv-row-command .kv-val,
    .kv-row-max-output-tokens .kv-val {
        font-family: var(--font-mono);
        font-size: 0.9rem;
        background: var(--code-bg);
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        border: 1px solid var(--border-color);
        display: inline-block;
    }
    
    .kv-row-cmd .kv-key,
    .kv-row-command .kv-key {
        display: none;
    }
    
    .kv-list .kv-list {
        margin-left: 0.75rem;
        padding-left: 0.75rem;
        border-left: 2px solid var(--border-subtle);
    }
    .kv-list .kv-list .kv-key {
        width: 120px;
    }

    /* Content Styling */
    pre, .code-block pre {
      background: var(--code-bg);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-default);
      padding: 1.25rem;
      overflow-x: auto;
      font-family: var(--font-mono);
      font-size: 0.95rem;
      line-height: 1.6;
      color: var(--text-primary);
      margin: 0.75rem 0;
    }
    
    code {
        font-family: var(--font-mono);
        background: rgba(110,118,129,0.4);
        padding: 0.2em 0.4em;
        border-radius: 4px;
        font-size: 85%;
    }
    pre code { background: none; padding: 0; font-size: 100%; }

    .code-block {
        position: relative;
        margin: 1rem 0;
    }
    .code-lang {
        position: absolute;
        top: 0;
        right: 0;
        background: var(--border-color);
        color: var(--text-secondary);
        padding: 0.2rem 0.6rem;
        font-size: 0.7rem;
        border-bottom-left-radius: var(--radius-default);
        border-top-right-radius: var(--radius-default); /* Match pre radius */
        z-index: 5;
    }

    .copy-btn {
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        z-index: 10;
        opacity: 0;
        transition: opacity 0.2s;
        background: var(--bg-panel);
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
    }
    .code-block:hover .copy-btn { opacity: 1; }
    .copy-btn:hover { color: var(--text-primary); border-color: var(--text-secondary); }

    /* Other elements */
    hr { border-top-color: var(--border-color); }
    
    .status-chip {
        border: 1px solid transparent;
        font-weight: 500;
    }
    .status-chip.status-completed {
        background: rgba(35, 134, 54, 0.2);
        color: #7ee787;
        border-color: rgba(35, 134, 54, 0.4);
    }
    .status-chip.status-in_progress {
        background: rgba(210, 153, 34, 0.2);
        color: #d29922;
        border-color: rgba(210, 153, 34, 0.4);
    }
    
    .kv-list {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        margin: 0.5rem 0;
    }
    .kv-row {
        display: flex;
        gap: 0.5rem;
        align-items: flex-start;
    }
    .kv-key {
        font-weight: 600;
        color: var(--text-secondary);
        font-size: 0.85rem;
        width: 140px;
        flex-shrink: 0;
    }
    .kv-key::after { content: ":"; }
    .kv-val {
        flex: 1;
        font-size: 0.85rem;
        min-width: 0;
    }
    .kv-row-cmd .kv-val,
    .kv-row-command .kv-val,
    .kv-row-max-output-tokens .kv-val {
        font-family: var(--font-mono);
        background: var(--code-bg);
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        border: 1px solid var(--border-subtle);
        display: inline-block;
        margin-top: -0.1rem;
    }
    
    .kv-list .kv-list {
        margin-left: 0.5rem;
        padding-left: 0.5rem;
        border-left: 1px solid var(--border-subtle);
    }
    .kv-list .kv-list .kv-key {
        width: 100px;
    }
    
    .plan-board {
        background: var(--bg-app);
        border: 1px solid var(--border-color);
    }

    /* Utility */
    .hidden { display: none !important; }
    .text-muted { color: var(--text-secondary); }
    
    /* Search highlighting */
    .highlight-match {
        background: rgba(210, 153, 34, 0.4);
        color: white;
    }

    /* Truncation */
    .truncated-details summary {
        list-style: none;
        user-select: none;
    }
    .truncated-details summary::-webkit-details-marker {
        display: none;
    }
    .truncated-details summary:hover {
        color: var(--accent-color);
    }

    /* Hiding logic */
    body.events-hidden .entry.event-block { display: none; }
    body.token-usage-hidden .entry.token-usage-block { display: none; }
    body.turn-context-hidden .entry.turn-context-block { display: none; }
    body.meta-hidden .entry.collapsible-meta { display: none; }
    body.prefix-collapsed .entry.pre-third-user { display: none; }
    body.summary-only .entry { display: none; }
    body.summary-only .entry.summary-block { display: block; }
"""


def build_page(
    entries: List[Entry],
    source_path: Path,
    sessions: List[SessionLog],
    active_session: SessionLog,
) -> str:
    collapsed_count = prepare_entries_for_render(entries)
    card_fragments = [entry_to_html(entry) for entry in entries]
    total = len(entries)
    max_lineno = entries[-1].lineno if entries else 0
    timeline_html = render_tool_timeline(entries)
    cards_json = json.dumps(card_fragments).replace("</", "<\\/")
    session_selector_html = render_session_selector(
        sessions, active_session, "/index.html")
    run_code_href = f"/run_code_log.html?sid={quote(active_session.session_id, safe='')}"
    escaped_sid = html.escape(active_session.session_id)
    body_class_attr = ' class="prefix-collapsed"' if collapsed_count else ""
    fold_controls_html = ""
    if collapsed_count:
        fold_controls_html = (
            f'<button id="toggle-prefix" class="nav-button secondary" type="button" '
            f'data-collapsed-text="Unfold earlier ({collapsed_count})" '
            'data-expanded-text="Fold earlier">'
            f"Unfold earlier ({collapsed_count})"
            "</button>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Conversation Viewer</title>
  <style>
{BASE_CSS}
  </style>
</head>
<body{body_class_attr}>
  <header>
    <div class="header-top">
      <h1>Conversation Viewer</h1>
      <div class="header-actions">
        <button id="toggle-theme" class="meta-toggle" type="button">Theme</button>
        <a href="{run_code_href}" class="nav-button">View uploads</a>
        <button id="toggle-meta" class="meta-toggle" type="button">Meta</button>
      </div>
    </div>
    
    {session_selector_html}

    <div class="controls-row">
       <input type="text" id="entry-search" placeholder="Filter entries..." style="width: 300px;">
       
       <div style="margin-left:auto; display:flex; gap:1rem; align-items:center;">
          <label style="font-size:0.85rem; color:var(--text-secondary); display:flex; align-items:center; gap:0.35rem; cursor:pointer;">
            <input id="toggle-events" type="checkbox"> Show Events
          </label>
          <label style="font-size:0.85rem; color:var(--text-secondary); display:flex; align-items:center; gap:0.35rem; cursor:pointer;">
            <input id="toggle-token-usage" type="checkbox"> Show Tokens
          </label>
          <label style="font-size:0.85rem; color:var(--text-secondary); display:flex; align-items:center; gap:0.35rem; cursor:pointer;">
            <input id="toggle-turn-context" type="checkbox"> Show Turn Context
          </label>
          <label style="font-size:0.85rem; color:var(--text-secondary); display:flex; align-items:center; gap:0.35rem; cursor:pointer;">
            <input id="toggle-summary-only" type="checkbox"> 只看 Summary
          </label>
       </div>
       
       {fold_controls_html}
    </div>
    
    <div style="margin-top:0.5rem; font-size:0.8rem; color:var(--text-secondary);">
      {html.escape(active_session.rel_path)} · <span id="entry-total">{total}</span> entries · {html.escape(active_session.time_label)}
    </div>
  </header>
  {timeline_html}
  <div class="container" id="cards-container"></div>
  <script>
    const cardFragments = {cards_json};
    const activeSessionId = "{escaped_sid}";
    let latestRenderedLine = {max_lineno};
    let initialRenderDone = false;
    
    // Theme logic
    (() => {{
      const btn = document.getElementById('toggle-theme');
      const body = document.body;
      const savedTheme = localStorage.getItem('viewcodexlog-theme');
      
      const setTheme = (theme) => {{
        if (theme === 'light') {{
          body.classList.add('theme-light');
        }} else {{
          body.classList.remove('theme-light');
        }}
        localStorage.setItem('viewcodexlog-theme', theme);
      }};

      if (savedTheme) {{
        setTheme(savedTheme);
      }}

      btn?.addEventListener('click', () => {{
        const isLight = body.classList.contains('theme-light');
        setTheme(isLight ? 'dark' : 'light');
      }});
    }})();

    // Copy function
    window.copyCode = (btn) => {{
      const pre = btn.parentElement.querySelector('pre code') || btn.parentElement.querySelector('pre');
      if (!pre) return;
      const text = pre.innerText;
      navigator.clipboard.writeText(text).then(() => {{
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = originalText, 2000);
      }});
    }};

    // Search logic
    (() => {{
      const input = document.getElementById('entry-search');
      if(!input) return;
      input.addEventListener('input', (e) => {{
        const term = e.target.value.toLowerCase();
        document.querySelectorAll('.entry').forEach(entry => {{
            const text = entry.innerText.toLowerCase();
            if (text.includes(term)) {{
                entry.classList.remove('hidden-search');
            }} else {{
                entry.classList.add('hidden-search');
            }}
        }});
      }});
      // Add CSS for hidden-search dynamically
      const style = document.createElement('style');
      style.textContent = '.hidden-search {{ display: none !important; }}';
      document.head.appendChild(style);
    }})();

    (() => {{
      const container = document.getElementById("cards-container");
      if (!container) return;
      const entryTotal = document.getElementById("entry-total");
      const batchSize = 200;
      let index = 0;
      const scheduleBatch = (cb) => {{
        if (typeof window.requestIdleCallback === "function") {{
          window.requestIdleCallback(cb);
          return;
        }}
        setTimeout(cb, 0);
      }};
      const appendCards = (fragments) => {{
        if (!Array.isArray(fragments) || fragments.length === 0) return;
        const frag = document.createDocumentFragment();
        fragments.forEach((item) => {{
          const wrapper = document.createElement("div");
          wrapper.innerHTML = item;
          if (wrapper.firstElementChild) {{
            frag.appendChild(wrapper.firstElementChild);
          }}
        }});
        container.appendChild(frag);
        // Re-apply search if exists
        const searchInput = document.getElementById('entry-search');
        if (searchInput && searchInput.value) {{
            searchInput.dispatchEvent(new Event('input'));
        }}
      }};
      const appendBatch = () => {{
        const limit = Math.min(index + batchSize, cardFragments.length);
        const chunk = [];
        for (; index < limit; index++) {{
          chunk.push(cardFragments[index]);
        }}
        appendCards(chunk);
        if (entryTotal) {{
          entryTotal.textContent = String(container.children.length);
        }}
        if (index < cardFragments.length) {{
          scheduleBatch(appendBatch);
        }} else {{
          initialRenderDone = true;
        }}
      }};
      appendBatch();

      const pollForNewEntries = async () => {{
        if (!initialRenderDone) {{
          setTimeout(pollForNewEntries, 500);
          return;
        }}
        const params = new URLSearchParams();
        if (activeSessionId) {{
          params.set("sid", activeSessionId);
        }}
        params.set("since", String(latestRenderedLine));
        try {{
          const response = await fetch(`/api/entries?${{params.toString()}}`, {{
            cache: "no-store",
          }});
          if (!response.ok) {{
            setTimeout(pollForNewEntries, 2000);
            return;
          }}
          const payload = await response.json();
          if (payload && payload.reset) {{
            location.reload();
            return;
          }}
          if (payload && Array.isArray(payload.fragments) && payload.fragments.length > 0) {{
            appendCards(payload.fragments);
            if (entryTotal) {{
              entryTotal.textContent = String(container.children.length);
            }}
          }}
          if (payload && typeof payload.max_lineno === "number") {{
            latestRenderedLine = Math.max(latestRenderedLine, payload.max_lineno);
          }}
        }} catch (_err) {{
          // Ignore transient polling errors.
        }}
        setTimeout(pollForNewEntries, 2000);
      }};
      setTimeout(pollForNewEntries, 2000);
    }})();

    (() => {{
      const eventsToggle = document.getElementById("toggle-events");
      const tokenToggle = document.getElementById("toggle-token-usage");
      const turnContextToggle = document.getElementById("toggle-turn-context");
      const summaryOnlyToggle = document.getElementById("toggle-summary-only");
      if (!eventsToggle || !tokenToggle || !turnContextToggle || !summaryOnlyToggle) return;
      const update = () => {{
        document.body.classList.toggle("events-hidden", !eventsToggle.checked);
        document.body.classList.toggle("token-usage-hidden", !tokenToggle.checked);
        document.body.classList.toggle("turn-context-hidden", !turnContextToggle.checked);
        document.body.classList.toggle("summary-only", summaryOnlyToggle.checked);
      }};
      eventsToggle.addEventListener("change", update);
      tokenToggle.addEventListener("change", update);
      turnContextToggle.addEventListener("change", update);
      summaryOnlyToggle.addEventListener("change", update);
      update();
    }})();

    (() => {{
      const btn = document.getElementById("toggle-meta");
      if (!btn) return;
      let hidden = false;
      const update = () => {{
        document.body.classList.toggle("meta-hidden", hidden);
        btn.textContent = hidden ? "Show meta" : "Hide meta";
      }};
      btn.addEventListener("click", () => {{
        hidden = !hidden;
        update();
      }});
      update();
    }})();

    (() => {{
      const btn = document.getElementById("toggle-prefix");
      if (!btn) return;
      let collapsed = document.body.classList.contains("prefix-collapsed");
      const collapsedText = btn.dataset.collapsedText || "Unfold earlier";
      const expandedText = btn.dataset.expandedText || "Fold earlier";
      const update = () => {{
        document.body.classList.toggle("prefix-collapsed", collapsed);
        btn.textContent = collapsed ? collapsedText : expandedText;
      }};
      btn.addEventListener("click", () => {{
        collapsed = !collapsed;
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
    id_attr = f' id="{html.escape(entry.anchor_id)}"' if entry.anchor_id else ""
    
    jump_btn = ""
    next_btn = ""
    if entry.tool_name and entry.anchor_id:
        jump_btn = (
            f'<button type="button" class="nav-button secondary small jump-to-timeline" '
            f'data-target="{html.escape(entry.anchor_id)}">Timeline</button>'
        )
        if entry.next_tool_anchor:
            next_btn = (
                f'<button type="button" class="nav-button secondary small jump-to-next" '
                f'data-target="{html.escape(entry.next_tool_anchor)}">Next</button>'
            )

    return (
        f'<article class="entry {classes}"{id_attr}>'
        f'<header>'
        f'<div style="display:flex;align-items:center;">'
        f'<button type="button" class="entry-toggle" onclick="this.closest(\'.entry\').classList.toggle(\'collapsed\')"></button>'
        f'<strong>{html.escape(entry.label)}</strong>'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:0.5rem;margin-left:auto;">'
        f'{jump_btn}{next_btn}'
        f'<span class="text-muted" style="font-size:0.8rem;white-space:nowrap;">{html.escape(format_timestamp_for_display(entry.timestamp))} · L{entry.lineno}</span>'
        f'</div>'
        f'</header>'
        f'<div class="entry-body">{entry.body_html}</div>'
        f'</article>'
    )


def build_run_code_page(
    source_path: Path,
    sessions: List[SessionLog],
    active_session: SessionLog,
) -> str:
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
    session_selector_html = render_session_selector(
        sessions, active_session, "/run_code_log.html")
    back_href = f"/index.html?sid={quote(active_session.session_id, safe='')}"

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
        <a href="{back_href}" class="nav-button secondary">Back to entries</a>
      </div>
    </div>
    {session_selector_html}
    <p>Source: {html.escape(active_session.rel_path)} · {total} uploads · {html.escape(active_session.time_label)}</p>
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
            f"<td>{html.escape(format_timestamp_for_display(upload.timestamp))}</td>"
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
    sessions_provider: Callable[[], List[SessionLog]],
    index_builder: Callable[[List[SessionLog], SessionLog], str],
    run_code_builder: Callable[[List[SessionLog], SessionLog], str],
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            path = parsed.path or "/"
            params = parse_qs(parsed.query)
            requested_sid = params.get("sid", [None])[0]
            sessions = sessions_provider()
            session = resolve_session_or_default(sessions, requested_sid)
            if session is None:
                self.send_error(500, "No session logs available")
                return
            if path == "/api/entries":
                since_raw = params.get("since", ["0"])[0]
                try:
                    since_lineno = max(0, int(since_raw))
                except (TypeError, ValueError):
                    self.send_error(400, "Invalid 'since' parameter")
                    return

                entries = load_entries(session.path)
                prepare_entries_for_render(entries)
                max_lineno = entries[-1].lineno if entries else 0
                reset = since_lineno > max_lineno
                new_entries: List[Entry] = []
                if not reset:
                    new_entries = [entry for entry in entries if entry.lineno > since_lineno]

                payload = {
                    "fragments": [entry_to_html(entry) for entry in new_entries],
                    "max_lineno": max_lineno,
                    "reset": reset,
                }
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path in {"/", "/index.html"}:
                body_builder = index_builder
            elif path == "/run_code_log.html":
                body_builder = run_code_builder
            else:
                self.send_error(404, "Not Found")
                return
            body = body_builder(sessions, session).encode("utf-8")
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


def write_html_output(output_path: Path, index_html: str) -> None:
    if output_path.suffix.lower() == ".html" and not output_path.is_dir():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(index_html, encoding="utf-8")
        print(f"Wrote HTML to {output_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    index_path = output_path / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    print(f"Wrote HTML to {index_path}")


def main() -> None:
    args = parse_args()
    selected: Optional[SessionLog] = None
    sessions_provider: Callable[[], List[SessionLog]]
    if args.log:
        source_path = Path(args.log).expanduser().resolve()
        if not source_path.exists():
            print(f"Log file not found: {source_path}", file=sys.stderr)
            sys.exit(1)
        session = build_session_log(source_path, source_path.parent)
        if session is None:
            timestamp = session_time_from_filename(source_path) or ""
            session = SessionLog(
                session_id=hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:16],
                path=source_path,
                rel_path=source_path.name,
                timestamp=timestamp,
                time_label=format_session_time_label(timestamp),
                title=source_path.stem,
                summary=source_path.stem,
            )
        selected = session
        sessions_provider = lambda: [session]
    else:
        sessions_dir = Path(args.sessions_dir).expanduser().resolve()
        initial_sessions = discover_session_logs(sessions_dir)
        if not initial_sessions:
            print(
                f"No JSONL logs were found under: {sessions_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        selected = initial_sessions[0]

        def sessions_provider() -> List[SessionLog]:
            return discover_session_logs(sessions_dir)

    if selected is None:
        print("No session was selected", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        entries = load_entries(selected.path)
        if not entries:
            print(f"No entries were parsed from: {selected.path}", file=sys.stderr)
            sys.exit(1)
        write_html_output(
            output_path,
            build_page(entries, selected.path, [selected], selected),
        )
        return

    def page_builder(sessions: List[SessionLog], session: SessionLog) -> str:
        entries = load_entries(session.path)
        return build_page(entries, session.path, sessions, session)

    def run_code_builder(sessions: List[SessionLog], session: SessionLog) -> str:
        return build_run_code_page(session.path, sessions, session)

    start_server(args.port, sessions_provider, page_builder, run_code_builder)


if __name__ == "__main__":
    main()
