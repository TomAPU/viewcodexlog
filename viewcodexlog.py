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
            anchor_id=f"entry-{lineno}",
            tool_name=name,
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
          cb.checked = tool === "{TARGET_RUN_CODE_FN}";
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
        const pad = panel.offsetHeight + 12;
        document.body.style.paddingTop = `${{pad}}px`;
        const maxScroll = Math.max(0, totalWidth - scrollBox.clientWidth);
        scrollBox.scrollLeft = Math.min(scrollBox.scrollLeft, maxScroll);
      }};

      const hidePanel = () => {{
        panel.classList.add("hidden");
        showBtn?.classList.remove("hidden");
        document.body.style.paddingTop = `12px`;
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


BASE_CSS = """
    :root {
      --page-max-width: 1680px;
      --reading-width: 1080px;
      --page-gutter: clamp(14px, 2vw, 32px);
    }
    *,
    *::before,
    *::after {
      box-sizing: border-box;
    }
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      background: #f5f6f8;
      color: #1c1c1c;
    }
    header {
      padding: 1rem var(--page-gutter);
      background: #232f3e;
      color: white;
    }
    header > .header-top,
    header > .session-switcher,
    header > p,
    header > .visibility-controls-row {
      width: 100%;
      max-width: var(--page-max-width);
      margin-left: auto;
      margin-right: auto;
    }
    header > p {
      margin: 0.8rem auto 0;
      color: #d7dee9;
    }
    h1 {
      margin: 0;
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
    .visibility-controls {
      display: inline-flex;
      align-items: center;
      gap: 0.8rem;
      padding: 0.35rem 0.75rem;
      border: 1px solid rgba(255, 255, 255, 0.35);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
    }
    .visibility-toggle {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.86rem;
      color: #edf3ff;
      user-select: none;
      cursor: pointer;
    }
    .visibility-toggle input {
      margin: 0;
      accent-color: #ffc107;
    }
    .visibility-controls-row {
      margin-top: 0.55rem;
    }
    .session-switcher {
      display: flex;
      flex-direction: column;
      align-items: stretch;
      gap: 0.45rem;
      margin-top: 0.65rem;
    }
    .session-switcher label {
      font-weight: 600;
      font-size: 0.92rem;
      color: #edf3ff;
    }
    .session-search-input {
      min-width: min(100%, 360px);
      width: min(100%, 760px);
      max-width: 100%;
      border-radius: 7px;
      border: 1px solid #cfd7e3;
      padding: 0.4rem 0.55rem;
      font-size: 0.9rem;
      background: white;
      color: #1c1c1c;
    }
    .session-list {
      list-style: none;
      margin: 0.15rem 0 0;
      padding: 0;
      width: min(100%, 1100px);
      max-height: 230px;
      overflow-y: auto;
      border: 1px solid #445267;
      border-radius: 8px;
      background: rgba(10, 17, 29, 0.35);
    }
    .session-item.hidden {
      display: none;
    }
    .session-item + .session-item {
      border-top: 1px solid rgba(215, 222, 233, 0.18);
    }
    .session-link {
      display: block;
      padding: 0.5rem 0.65rem;
      color: #edf3ff;
      text-decoration: none;
    }
    .session-link:hover {
      background: rgba(255, 255, 255, 0.08);
    }
    .session-item.is-active .session-link {
      background: rgba(13, 110, 253, 0.25);
      border-left: 3px solid #6ea8fe;
      padding-left: calc(0.65rem - 3px);
    }
    .session-title {
      font-size: 0.88rem;
      line-height: 1.35;
      color: #f4f8ff;
    }
    .session-meta {
      margin-top: 0.18rem;
      font-size: 0.78rem;
      color: #bec8d8;
      line-height: 1.3;
    }
    .session-switcher .session-path {
      color: #b7c0cf;
      font-size: 0.8rem;
    }
    .container {
      padding: 1rem var(--page-gutter) 3rem;
      width: 100%;
      max-width: var(--page-max-width);
      margin: 0 auto;
    }
    .entry {
      background: white;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      padding: 1rem;
      width: min(100%, var(--reading-width));
      margin-left: auto;
      margin-right: auto;
      margin-bottom: 1rem;
      border-left: 4px solid transparent;
    }
    .entry header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 0.35rem 0.6rem;
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
    .nav-button.small {
      padding: 0.3rem 0.6rem;
      font-size: 0.85rem;
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
    body.events-hidden .entry.event-block {
      display: none;
    }
    body.token-usage-hidden .entry.token-usage-block {
      display: none;
    }
    body.meta-hidden .entry.collapsible-meta {
      display: none;
    }
    .panel {
      background: white;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      padding: 1rem;
      width: min(100%, var(--reading-width));
      margin-left: auto;
      margin-right: auto;
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
      width: min(100%, var(--reading-width));
      margin-left: auto;
      margin-right: auto;
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
    .timeline-panel {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      width: 100%;
      background: white;
      border-radius: 0 0 12px 12px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.12);
      padding: 0.85rem var(--page-gutter) 1rem;
      z-index: 30;
      border-bottom: 1px solid #e5e7eb;
    }
    .timeline-inner {
      width: 100%;
      max-width: var(--page-max-width);
      margin: 0 auto;
    }
    .timeline-panel.hidden { display: none; }
    .timeline-header {
      display: flex;
      align-items: flex-start;
      justify-content: flex-start;
      gap: 0.5rem 0.75rem;
      flex-wrap: wrap;
      width: 100%;
    }
    .timeline-range {
      color: #6c757d;
      font-size: 0.85rem;
      margin-top: 0.1rem;
    }
    .timeline-scroll {
      position: relative;
      overflow-x: auto;
      overflow-y: hidden;
      padding-bottom: 0.25rem;
      margin-top: 0.65rem;
    }
    .timeline-scroll::-webkit-scrollbar {
      height: 10px;
    }
    .timeline-scroll::-webkit-scrollbar-thumb {
      background: #cbd5e1;
      border-radius: 999px;
    }
    .timeline-scroll::-webkit-scrollbar-track {
      background: #edf2f7;
    }
    .timeline-track {
      position: relative;
      min-height: 64px;
      background: #f3f4f6;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      transition: height 0.15s ease;
    }
    .timeline-event {
      position: absolute;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      border: 2px solid #ffffff;
      box-shadow: 0 2px 6px rgba(0,0,0,0.2);
      cursor: pointer;
      transform: translateX(-50%);
      transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .timeline-event:hover {
      transform: translateX(-50%) scale(1.08);
      box-shadow: 0 4px 10px rgba(0,0,0,0.28);
    }
    .timeline-axis {
      position: relative;
      margin-top: 0.35rem;
      height: 22px;
    }
    .timeline-tick {
      position: absolute;
      top: 0;
      transform: translateX(-50%);
      color: #6c757d;
      font-size: 0.75rem;
      text-align: center;
      white-space: nowrap;
    }
    .timeline-tick::before {
      content: "";
      display: block;
      width: 1px;
      height: 8px;
      background: #ced4da;
      margin: 0 auto 2px;
    }
    .timeline-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem 0.75rem;
      margin-top: 0.6rem;
      font-size: 0.85rem;
      color: #495057;
    }
    .timeline-legend .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.1rem 0.35rem;
      border-radius: 6px;
      background: #f8f9fb;
      border: 1px solid #e3e7ed;
      cursor: pointer;
      user-select: none;
    }
    .timeline-legend input[type="checkbox"] {
      margin: 0;
      cursor: pointer;
    }
    .timeline-legend .swatch {
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
      border: 1px solid rgba(0,0,0,0.08);
    }
    .legend-actions {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-top: 0.5rem;
    }
    .timeline-toggle-btn {
      position: fixed;
      top: 0.75rem;
      right: max(var(--page-gutter), calc((100vw - var(--page-max-width)) / 2 + var(--page-gutter)));
      background: #0d6efd;
      color: white;
      border: none;
      border-radius: 999px;
      padding: 0.6rem 0.9rem;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 8px 18px rgba(13,110,253,0.25);
      z-index: 25;
    }
    .timeline-toggle-btn.hidden { display: none; }
    .timeline-event.highlight {
      box-shadow: 0 0 0 4px #ffd166, 0 3px 10px rgba(0,0,0,0.32);
      transform: translateX(-50%) scale(1.1);
    }
    .timeline-header > button {
      flex-shrink: 0;
    }
    .timeline-event.highlight {
      box-shadow: 0 0 0 4px #ffd166, 0 3px 10px rgba(0,0,0,0.32);
      transform: translateX(-50%) scale(1.1);
    }
    .entry-highlight {
      box-shadow: 0 0 0 3px #ffd166 inset, 0 6px 16px rgba(0,0,0,0.16);
      transition: box-shadow 0.3s ease;
    }
    @media (min-width: 1800px) {
      :root {
        --reading-width: 1160px;
      }
      .entry,
      .panel,
      .diff-card {
        padding: 1.1rem 1.2rem;
      }
    }
    @media (max-width: 640px) {
      .timeline-panel {
        padding: 0.75rem var(--page-gutter) 0.85rem;
        border-radius: 0 0 10px 10px;
      }
      .timeline-toggle-btn {
        right: max(var(--page-gutter), calc((100vw - var(--page-max-width)) / 2 + var(--page-gutter)));
        top: 0.65rem;
      }
    }
"""


def build_page(
    entries: List[Entry],
    source_path: Path,
    sessions: List[SessionLog],
    active_session: SessionLog,
) -> str:
    next_for_tool: dict[str, Optional[str]] = {}
    for entry in reversed(entries):
        if entry.tool_name and entry.anchor_id:
            entry.next_tool_anchor = next_for_tool.get(entry.tool_name)
            next_for_tool[entry.tool_name] = entry.anchor_id
    card_fragments = [entry_to_html(entry) for entry in entries]
    total = len(entries)
    timeline_html = render_tool_timeline(entries)
    cards_json = json.dumps(card_fragments).replace("</", "<\\/")
    session_selector_html = render_session_selector(
        sessions, active_session, "/index.html")
    run_code_href = f"/run_code_log.html?sid={quote(active_session.session_id, safe='')}"
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
        <a href="{run_code_href}" class="nav-button">View run_code uploads</a>
        <button id="toggle-meta" class="meta-toggle" type="button">Hide meta blocks</button>
      </div>
    </div>
    {session_selector_html}
    <p>Source: {html.escape(active_session.rel_path)} · {total} entries · {html.escape(active_session.time_label)}</p>
    <div class="visibility-controls-row">
      <div class="visibility-controls">
        <label class="visibility-toggle">
          <input id="toggle-events" type="checkbox">
          <span>Event</span>
        </label>
        <label class="visibility-toggle">
          <input id="toggle-token-usage" type="checkbox">
          <span>Token usage</span>
        </label>
      </div>
    </div>
  </header>
  {timeline_html}
  <div class="container" id="cards-container"></div>
  <script>
    const cardFragments = {cards_json};
    (() => {{
      const container = document.getElementById("cards-container");
      if (!container) return;
      const batchSize = 200;
      let index = 0;
      const appendBatch = () => {{
        const limit = Math.min(index + batchSize, cardFragments.length);
        const frag = document.createDocumentFragment();
        for (; index < limit; index++) {{
          const wrapper = document.createElement("div");
          wrapper.innerHTML = cardFragments[index];
          frag.appendChild(wrapper.firstElementChild);
        }}
        container.appendChild(frag);
        if (index < cardFragments.length) {{
          requestIdleCallback(appendBatch);
        }}
      }};
      appendBatch();
    }})();

    (() => {{
      const eventsToggle = document.getElementById("toggle-events");
      const tokenToggle = document.getElementById("toggle-token-usage");
      if (!eventsToggle || !tokenToggle) return;
      const update = () => {{
        document.body.classList.toggle("events-hidden", !eventsToggle.checked);
        document.body.classList.toggle("token-usage-hidden", !tokenToggle.checked);
      }};
      eventsToggle.addEventListener("change", update);
      tokenToggle.addEventListener("change", update);
      update();
    }})();

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
    id_attr = f' id="{html.escape(entry.anchor_id)}"' if entry.anchor_id else ""
    jump_btn = ""
    next_btn = ""
    if entry.tool_name and entry.anchor_id:
        jump_btn = (
            f'<button type="button" class="nav-button secondary small jump-to-timeline" '
            f'data-target="{html.escape(entry.anchor_id)}">On timeline</button>'
        )
        if entry.next_tool_anchor:
            next_btn = (
                f'<button type="button" class="nav-button secondary small jump-to-next" '
                f'data-target="{html.escape(entry.next_tool_anchor)}">Next this tool</button>'
            )
    return (
        f'<article class="entry {classes}"{id_attr}>'
        f"<header>"
        f"<div>{html.escape(entry.label)}</div>"
        f"<small>{html.escape(format_timestamp_for_display(entry.timestamp))} · line {entry.lineno} · {html.escape(entry.raw_type)}</small>"
        f"{jump_btn}{next_btn}"
        f"</header>"
        f"<div>{entry.body_html}</div>"
        f"</article>"
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
    sessions: List[SessionLog],
    index_builder: Callable[[SessionLog], str],
    run_code_builder: Callable[[SessionLog], str],
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            path = parsed.path or "/"
            params = parse_qs(parsed.query)
            requested_sid = params.get("sid", [None])[0]
            session = resolve_session_or_default(sessions, requested_sid)
            if session is None:
                self.send_error(500, "No session logs available")
                return
            if path in {"/", "/index.html"}:
                body_builder = index_builder
            elif path == "/run_code_log.html":
                body_builder = run_code_builder
            else:
                self.send_error(404, "Not Found")
                return
            body = body_builder(session).encode("utf-8")
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
    sessions: List[SessionLog] = []
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
        sessions = [session]
    else:
        sessions_dir = Path(args.sessions_dir).expanduser().resolve()
        sessions = discover_session_logs(sessions_dir)
        if not sessions:
            print(
                f"No JSONL logs were found under: {sessions_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

    selected = sessions[0]

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

    def page_builder(session: SessionLog) -> str:
        entries = load_entries(session.path)
        return build_page(entries, session.path, sessions, session)

    def run_code_builder(session: SessionLog) -> str:
        return build_run_code_page(session.path, sessions, session)

    start_server(args.port, sessions, page_builder, run_code_builder)


if __name__ == "__main__":
    main()
