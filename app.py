import re
import httpx
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import OpenAI
from duckduckgo_search import DDGS
from datetime import datetime
from typing import AsyncGenerator, Optional
import wikipediaapi
import sqlite3
import os
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import asyncio

# ----------------------------------------------------
# LOGGING
# ----------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# FASTAPI SETUP
# ----------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# DATABASE SETUP
# ----------------------------------------------------
DB_PATH = "data.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                is_web_search INTEGER DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

init_db()

def load_conversation(session_id: str) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content, is_web_search FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp
        """, (session_id,))
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "is_web_search": bool(row["is_web_search"])
            }
            for row in cursor.fetchall()
        ]


def save_message(session_id: str, role: str, content: str, is_web_search: int = 0):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, role, content, is_web_search) VALUES (?, ?, ?, ?)",
            (session_id, role, content, is_web_search)
        )
        conn.commit()

# ----------------------------------------------------
# MODELS
# ----------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2048)
    session_id: Optional[str] = None

# ----------------------------------------------------
# LLM CLIENT
# ----------------------------------------------------
client = OpenAI(
    base_url="https://nova-docker-app.onrender.com/engines/llama.cpp/v1",
    api_key="not-needed"
)

wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="NOVA/1.0 (jayanthkopparthi595@gmail.com)"
)

# ----------------------------------------------------
# SYSTEM PROMPTS
# ----------------------------------------------------
CHAT_SYSTEM_PROMPT = """You are NOVA ... (same text here)"""

WEB_SEARCH_SYSTEM_PROMPT = """You are NOVA ... (same text here)"""

# ----------------------------------------------------
# URL CHECKER
# ----------------------------------------------------
url_pattern = re.compile(r'https?://[\w\-./?=&%]+', re.IGNORECASE)

async def check_url(url: str) -> str:
    if url.lower().startswith("javascript:"):
        return f"{url} (unsafe)"
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "NOVA/1.0"})
            return f"{url} (accessible)" if r.status_code == 200 else f"{url} (status {r.status_code})"
    except Exception as e:
        return f"{url} ({e.__class__.__name__})"

# ----------------------------------------------------
# HEARTBEAT (prevents Render from killing SSE)
# ----------------------------------------------------
async def heartbeat():
    while True:
        yield "data: [ping]\n\n"
        await asyncio.sleep(10)

# ----------------------------------------------------
# MAIN CHAT STREAMING GENERATOR
# ----------------------------------------------------
async def generate_chat_response(session_id: str, message: str, system_prompt: str, is_web_search: int = 0):

    try:
        accumulated_response = ""
        messages = [{"role": "system", "content": system_prompt}] + \
                   load_conversation(session_id) + \
                   [{"role": "user", "content": message}]

        stream = client.chat.completions.create(
            model="ai/gemma3",
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=2048
        )

        word_buffer = ""

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                accumulated_response += content

                for char in content:
                    word_buffer += char

                    if char in [' ', '\n', '.', '!', '?', ';', ':', ',']:

                        yield f"data: {word_buffer.strip()}\n\n"
                        word_buffer = ""

        # final words
        if word_buffer.strip():
            yield f"data: {word_buffer}\n\n"

        # URL checking
        urls = url_pattern.findall(accumulated_response)
        if urls:
            yield "data: ## Sources\n\n"
            for url in urls:
                yield f"data: - {await check_url(url)}\n\n"

        save_message(session_id, "user", message, is_web_search)
        save_message(session_id, "assistant", accumulated_response, is_web_search)

    except Exception as e:
        error_msg = f"## Error\n\n{str(e)}"
        logger.error(error_msg)
        yield f"data: {error_msg}\n\n"

# ----------------------------------------------------
# SSE HEADERS (critical for Render)
# ----------------------------------------------------
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "text/event-stream",
    "X-Accel-Buffering": "no"
}

# ----------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------

@app.post("/chat")
async def chat_new(request: ChatRequest):

    session_id = f"session_{int(datetime.now().timestamp())}_{os.urandom(4).hex()}"
    current_time = datetime.now().strftime("%B %d, %Y, %I:%M %p")

    system_prompt = CHAT_SYSTEM_PROMPT.format(current_time=current_time)

    async def combined():
        async for part in generate_chat_response(session_id, request.message, system_prompt, 0):
            yield part
        async for ping in heartbeat():
            yield ping

    return StreamingResponse(combined(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/chat/{session_id}")
async def chat_continue(request: ChatRequest, session_id: str):

    current_time = datetime.now().strftime("%B %d, %Y, %I:%M %p")
    system_prompt = CHAT_SYSTEM_PROMPT.format(current_time=current_time)

    async def combined():
        async for part in generate_chat_response(session_id, request.message, system_prompt, 0):
            yield part
        async for ping in heartbeat():
            yield ping

    return StreamingResponse(combined(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/web_search")
async def web_search(request: ChatRequest):

    query = request.message
    session_id = request.session_id or f"search_{int(datetime.now().timestamp())}_{os.urandom(4).hex()}"
    current_time = datetime.now().strftime("%B %d, %Y, %I:%M %p")

    async def combined():
        async for part in generate_chat_response(session_id, query, WEB_SEARCH_SYSTEM_PROMPT.format(current_time=current_time), 1):
            yield part
        async for ping in heartbeat():
            yield ping

    return StreamingResponse(combined(), media_type="text/event-stream", headers=SSE_HEADERS)


# ----------------------------------------------------
# SESSION MANAGEMENT
# ----------------------------------------------------
@app.get("/sessions")
async def list_sessions():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT session_id, MAX(timestamp) as last_message
            FROM conversations
            GROUP BY session_id
            ORDER BY last_message DESC
        """)
        sessions = []
        for row in cur.fetchall():
            cur.execute("""
                SELECT content FROM conversations
                WHERE session_id = ? AND role = 'user'
                ORDER BY timestamp LIMIT 1
            """, (row["session_id"],))
            first = cur.fetchone()
            title = (first["content"][:50] + "...") if first and len(first["content"]) > 50 else (first["content"] if first else "Untitled")

            sessions.append({
                "session_id": row["session_id"],
                "title": title,
                "last_message": row["last_message"]
            })
    return {"sessions": sessions}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT role, content, timestamp, is_web_search
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp
        """, (session_id,))
        rows = cur.fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": session_id,
            "messages": [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                    "is_web_search": bool(r["is_web_search"])
                }
                for r in rows
            ]
        }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM conversations WHERE session_id = ?", (session_id,))
        cnt = cur.fetchone()[0]
        if cnt == 0:
            raise HTTPException(status_code=404, detail="Session not found")

        cur.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        conn.commit()

    return {"message": f"Session {session_id} deleted successfully"}


# ----------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------
@app.get("/")
async def root():
    return {"message": "NOVA Backend running", "status": "ok"}

