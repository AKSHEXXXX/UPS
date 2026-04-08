# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fastapi.responses import HTMLResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError("openenv is required for the web interface. Install dependencies with 'uv sync'") from e

from core.models import ColdChainAction, ColdChainObservation
from .environment import ColdChainEnvironment


app = create_app(
    ColdChainEnvironment,
    ColdChainAction,
    ColdChainObservation,
    env_name="coldchain-gym",
    max_concurrent_envs=1,
)


@app.get("/")
def root() -> dict:
        """Lightweight route index for local development and smoke checks."""
        return {
                "name": "coldchain-gym",
                "status": "healthy",
                "endpoints": {
                        "web": "/web",
                        "docs": "/docs",
                        "health": "/health",
                        "ws": "/ws",
                        "reset": "/reset",
                        "step": "/step",
                        "state": "/state",
                        "metadata": "/metadata",
                        "schema": "/schema",
                        "mcp": "/mcp",
                },
        }


@app.get("/web", response_class=HTMLResponse)
def web() -> str:
        """Human-friendly landing page with key OpenEnv links."""
        return """
<!doctype html>
<html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>coldchain-gym</title>
        <style>
            :root {
                --bg: #f6f8fb;
                --panel: #ffffff;
                --text: #10233d;
                --muted: #47607f;
                --accent: #0c8a6a;
            }
            body {
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: radial-gradient(circle at 0% 0%, #e8fff6, transparent 45%), var(--bg);
                color: var(--text);
            }
            .wrap {
                max-width: 760px;
                margin: 40px auto;
                padding: 24px;
            }
            .card {
                background: var(--panel);
                border: 1px solid #d7e0ea;
                border-radius: 14px;
                padding: 20px;
                box-shadow: 0 10px 25px rgba(20, 41, 72, 0.06);
            }
            h1 { margin-top: 0; }
            p { color: var(--muted); }
            ul { padding-left: 18px; }
            a { color: var(--accent); text-decoration: none; }
            a:hover { text-decoration: underline; }
            code {
                background: #eef4f9;
                padding: 2px 6px;
                border-radius: 6px;
            }
        </style>
    </head>
    <body>
        <main class=\"wrap\">
            <section class=\"card\">
                <h1>coldchain-gym is running</h1>
                <p>This OpenEnv environment exposes the standard interface and diagnostics endpoints.</p>
                <ul>
                    <li><a href=\"/docs\">API Docs</a></li>
                    <li><a href=\"/health\">Health</a></li>
                    <li><a href=\"/metadata\">Metadata</a></li>
                    <li><a href=\"/schema\">Schema</a></li>
                </ul>
                <p>Client interface: <code>reset()</code>, <code>step()</code>, and <code>state()</code> over <code>/ws</code>.</p>
            </section>
        </main>
    </body>
</html>
"""


def main() -> None:
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn server.app:app --workers 4
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
