# View Codex Log

> This entire project was crafted by Codex, and its functionality has been thoroughly cat-tested. 🐾

`viewcodexlog.py` is a tiny Python web app that renders Codex CLI JSONL traces as a readable conversation timeline. Point it at any `*.jsonl` log and it spins up a local HTTP server with an interactive HTML view.

## Features

- **Conversation cards** for every `response_item`, `event_msg`, `session_meta`, and `turn_context`.
- **Smart formatting** for function calls/outputs, including tables for structured data, code blocks for `{"type":"code"}` nodes, and a custom board for `update_plan`.
- **Collapsible metadata** so heavy JSON blobs (turn context, token usage, reasoning notes) stay out of the way. A header toggle hides/shows them all at once.
- **Works offline**: no dependencies beyond the Python standard library.

## Quick start

```bash
python3 viewcodexlog.py -l kctf_poc_gen_success.jsonl -p 8123
```

Then open `http://127.0.0.1:8123` in your browser. Use the “Hide meta blocks” button if you want to focus on user/assistant/tool turns.

## Command reference

| Flag | Description |
| ---- | ----------- |
| `-l, --log` | Path to the JSONL log (required). |
| `-p, --port` | Port for the HTTP server (default `8000`). |

## Development

- Python 3.8+ recommended (uses only stdlib).
- No external dependencies; running `python3 -m py_compile viewcodexlog.py` is enough for a quick lint.
- The script currently loads the entire log into memory. For very large logs consider streaming/pagination if needed.

Contributions are welcome—file an issue or PR once this lands on GitHub!
