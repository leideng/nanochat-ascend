# Web Demo

## Purpose

This page shows the exact startup flow for the browser chat UI in `nanochat/ui.html`.

The UI depends on backend API endpoints (`/health` and `/chat/completions`), so it must be served through `scripts/chat_web.py` rather than opened directly from `file://...`.

## Quick Start

From the repository root:

```bash
source runs/set_env.sh
uv sync
source .venv/bin/activate
python -m scripts.chat_web --device-type=cpu --host 127.0.0.1 --port 8000
```

Then open:

- <http://127.0.0.1:8000>

## Choose Model Source And Checkpoint

If you want a specific source/checkpoint, pass it explicitly:

```bash
python -m scripts.chat_web --device-type=cpu --source rl --model-tag d20
```

Useful arguments:

- `--source`: `sft` or `rl`
- `--model-tag`: model family tag (for example `d20`)
- `--step`: specific checkpoint step
- `--num-devices`: number of devices to load (NPU multi-device scenario)
- `--port`: server port (default `8000`)

## How It Connects

`nanochat/ui.html` uses relative URLs:

- `GET /health` for engine readiness
- `POST /chat/completions` for streaming generation

Because `API_URL` is empty in the page script, the browser calls the same origin that served the HTML. Running `scripts/chat_web.py` provides both the page and API on that same origin.

## Troubleshooting

### "Engine not running. Please start engine.py first."

This message usually means the browser could not reach `/health`.

Check:

1. The server process is running and listening on the same host/port you opened in the browser.
2. You opened `http://127.0.0.1:8000` (or your configured host/port), not `file:///.../ui.html`.
3. The model checkpoint for your selected `--source`/`--model-tag` exists in your configured checkpoint directory.

### Port Already In Use

Run on another port:

```bash
python -m scripts.chat_web --device-type=cpu --host 127.0.0.1 --port 8010
```

Then open `http://127.0.0.1:8010`.
