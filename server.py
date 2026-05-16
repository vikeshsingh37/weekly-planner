#!/usr/bin/env python3
"""
Weekly Planner Agent — web UI entry point.

Usage:
    uv run python server.py              # http://localhost:8000
    uv run python server.py --port 3000

All API endpoints require a Bearer JWT obtained from /auth/login or
/auth/register. The WebSocket accepts the token as a query parameter
(?token=...) because browsers cannot set headers for WebSocket connections.
"""

import argparse
import asyncio
import os
from pathlib import Path
from typing import Annotated

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from agent.agent import WeeklyPlannerAgent
from api.models import UpdatePreferencesInput
from impl.auth import authenticate, create_token, email_to_user_id, register, verify_token
from impl.memory import JSONSessionManager
from impl.postgres_memory import PostgresSessionManager, ensure_table
from impl.tools import ToolRunner

_DATABASE_URL: str | None = os.environ.get("DATABASE_URL")

_TOOL_LABELS: dict[str, str] = {
    "parse_and_add_tasks": "Adding tasks",
    "schedule_tasks":      "Scheduling",
    "move_task":           "Moving task",
    "remove_task":         "Removing task",
    "get_schedule":        "Reading schedule",
    "update_preferences":  "Updating preferences",
    "get_weather":         "Checking weather",
}

app = FastAPI(title="Weekly Planner")

@app.on_event("startup")
def _startup():
    if _DATABASE_URL:
        ensure_table(_DATABASE_URL)
_HTML = (Path(__file__).parent / "static" / "index.html").read_text
_bearer = HTTPBearer(auto_error=False)


# ── Auth models ────────────────────────────────────────────────────────────────

class _AuthRequest(BaseModel):
    email: str
    password: str


# ── Auth dependency ────────────────────────────────────────────────────────────

def _require_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    token = creds.credentials if creds else ""
    user_id = verify_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user_id


def _make_session(user_id: str) -> JSONSessionManager | PostgresSessionManager:
    if _DATABASE_URL:
        return PostgresSessionManager(user_id=user_id, conninfo=_DATABASE_URL)
    path = f"sessions/{user_id}"
    os.makedirs(path, exist_ok=True)
    return JSONSessionManager(session_file=f"{path}/state.json")


# ── Static ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return HTMLResponse(_HTML())


# ── Auth endpoints (public) ────────────────────────────────────────────────────

@app.post("/auth/register")
async def auth_register(body: _AuthRequest):
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    ok = register(body.email, body.password)
    if not ok:
        raise HTTPException(status_code=409, detail="Email already registered")
    token = create_token(body.email)
    return JSONResponse({"token": token, "user_id": email_to_user_id(body.email)})


@app.post("/auth/login")
async def auth_login(body: _AuthRequest):
    user_id = authenticate(body.email, body.password)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(body.email)
    return JSONResponse({"token": token, "user_id": user_id})


# ── Protected API endpoints ────────────────────────────────────────────────────

@app.get("/api/schedule")
async def api_get_schedule(user_id: Annotated[str, Depends(_require_user)]):
    mgr = _make_session(user_id)
    state = mgr.state.model_dump(exclude={"conversation_history"})
    return JSONResponse(state)


@app.get("/api/location")
async def api_get_location(request: Request):
    """Detect approximate location from the client's IP — no auth required."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            url = (
                "https://ipapi.co/json/"
                if client_ip in ("127.0.0.1", "::1")
                else f"https://ipapi.co/{client_ip}/json/"
            )
            r = await client.get(url)
            if r.status_code == 200:
                d = r.json()
                if not d.get("error"):
                    name_parts = [p for p in [d.get("city", ""), d.get("region", ""), d.get("country_code", "")] if p]
                    return JSONResponse({
                        "latitude": d.get("latitude"),
                        "longitude": d.get("longitude"),
                        "location_name": ", ".join(name_parts),
                    })
    except Exception:
        pass
    return JSONResponse({"error": "Could not detect location"})


@app.post("/api/preferences")
async def api_save_preferences(
    body: UpdatePreferencesInput,
    user_id: Annotated[str, Depends(_require_user)],
):
    mgr = _make_session(user_id)
    try:
        mgr.update_preferences(body.model_dump(exclude_none=True))
        mgr.save()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(mgr.state.preferences.model_dump())


# ── WebSocket (token via query param — browsers can't set WS headers) ──────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, token: str = Query(default="")):
    user_id = verify_token(token)
    if not user_id:
        await websocket.close(code=4401, reason="Unauthorized")
        return

    await websocket.accept()

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    mgr = _make_session(user_id)

    def on_event(event: str, *args):
        payload: dict = {"type": event}
        if event in ("tool_start", "tool_end"):
            name = args[0]
            payload["name"] = name
            payload["label"] = _TOOL_LABELS.get(name, name)
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    agent = WeeklyPlannerAgent(
        session=mgr, tools=ToolRunner(), on_event=on_event, user_id=user_id
    )

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") != "message":
                continue
            user_text = data.get("text", "").strip()
            if not user_text:
                continue

            captured = user_text

            async def run_agent(text: str = captured):
                try:
                    response = await loop.run_in_executor(None, agent.chat, text)
                    queue.put_nowait({"type": "response", "text": response})
                except Exception as exc:
                    queue.put_nowait({"type": "error", "text": str(exc)})
                finally:
                    queue.put_nowait({"type": "done"})

            asyncio.create_task(run_agent())

            while True:
                event = await queue.get()
                await websocket.send_json(event)
                if event["type"] in ("done", "error"):
                    break

    except WebSocketDisconnect:
        pass


def main():
    parser = argparse.ArgumentParser(description="Weekly Planner web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    url = f"http://{args.host}:{args.port}"
    print(f"Weekly Planner → {url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()