# View Codex Log

**Batch Viewer for Codex CLI JSONL Logs**

> This entire project was crafted by Codex, and its functionality has been thoroughly cat-tested. 🐾

`viewcodexlog.py` is a tiny Python web app that renders Codex CLI JSONL traces as a readable conversation timeline. Point it at any `*.jsonl` log and it spins up a local HTTP server with an interactive HTML view.

## Quick start

```bash
python viewcodexlog.py -l kernelmcp-cleaned-experiments\kctf-poc-gen\cases -p 8123
python viewcodexlog.py -l kctf\mass-run-medium -p 8123
```

Then open `http://127.0.0.1:8123` in your browser. Use the “Hide meta blocks” button if you want to focus on user/assistant/tool turns.

## Command reference

| Flag | Description |
| ---- | ----------- |
| `-l, --log` | Path to the folder containing (hash/sessions/yyyy/mm/dd/rollout*.jsonl) JSONL logs (required). |
| `-p, --port` | Port for the HTTP server (default `8000`). |
