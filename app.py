import re
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import OpenAI
from duckduckgo_search import DDGS
from datetime import datetime
from typing import AsyncGenerator
import wikipediaapi
import sqlite3
import os
from fastapi.middleware.cors import CORSMiddleware
import logging
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Setup
DB_PATH = "data.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
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
            SELECT role, content FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp
        """, (session_id,))
        return [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]

def save_message(session_id: str, role: str, content: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()

# Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2048)
    session_id: str = None

# OpenAI Client to Local LLM
client = OpenAI(
    base_url="https://nova-docker-app.onrender.com",
    api_key="not-needed"
)

# Wikipedia API Client
wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="NOVA/1.0 (jayanthkopparthi595@gmail.com)"
)

# Assistant Persona
CHAT_SYSTEM_PROMPT = """You are NOVA, a knowledgeable assistant specializing in clear, structured responses. For all queries, especially mathematical ones, respond in markdown format with:
- **Headings** (`##`, `###`) for main topics and subtopics.
- **Bullet points** (`-`) or **numbered lists** (`1.`, `2.`) for key points or steps.
- **LaTeX** for equations (e.g., `$E = mc^2$` for inline, `$$E = mc^2$$` for display).
- **Tables** (`| Header | Header |`) for data or comparisons when relevant.
- For math queries (e.g., integrals, derivatives):
  - Include a **Definition** section with the concept explained clearly.
  - Use **LaTeX** for all mathematical expressions (e.g., `$$\\int_a^b f(x) \\, dx$$`).
  - Provide an **Example** section with a sample problem and solution.
- For informational queries:
  - Include a **Key Points** section with bullet points or numbered lists.
  - Add a **Details** section for deeper explanation if applicable.
- For task-oriented queries:
  - Provide a **Steps** section with numbered steps.
  - Include a **Tips** section if relevant.
- Use single newlines between paragraphs and list items unless a blank line is needed for markdown (e.g., before headings or tables).
- Avoid bold (`**`) for inline text unless emphasizing a term; use it for headings or list prefixes sparingly.
- If URLs are included, list them in a **Sources** section with markdown links (`[Source](URL)`).
- Never invent facts. If unsure, state: "I don't have enough information to answer fully."
- Current time: {current_time}.
"""
WEB_SEARCH_SYSTEM_PROMPT = """You are NOVA, an expert research assistant. Respond in markdown format with:
- **Headings** (`##`, `###`) for main topics and subtopics.
- **Key Facts** section with bullet points (`-`) or numbered lists (`1.`, `2.`).
- **LaTeX** for equations (e.g., `$E = mc^2$` for inline, `$$E = mc^2$$` for display).
- **Tables** (`| Header | Header |`) for comparisons or data.
- **Sources** section with markdown links (`[Source](URL)`).
- For math-related queries:
  - Include a **Definition** section with LaTeX equations.
  - Provide an **Example** section with a sample problem.
- Summarize information from provided sources (e.g., web results, Wikipedia) concisely.
- Cite sources clearly (e.g., "According to Wikipedia...").
- If sources conflict, include a **Discrepancies** section.
- If no reliable sources are found, state: "No reliable information found for '{query}' as of {current_time}."
- Use single newlines between paragraphs and list items unless a blank line is needed for markdown.
- Verify URLs for accessibility.
- Never invent facts or sources. If data is incomplete, note it.
- Current time: {current_time}.
"""

# URL Regex & Checker
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

def format_sse_data(content: str) -> str:
    """Format content for Server-Sent Events, handling newlines properly"""
    if not content:
        return ""
    lines = content.split('\n')
    formatted_lines = []
    for line in lines:
        escaped_line = line.replace('data: ', 'data\\: ')
        formatted_lines.append(f"data: {escaped_line}")
    return '\n'.join(formatted_lines) + '\n\n'

def send_chunk_properly(content: str) -> str:
    """Send a chunk with proper SSE formatting and newline handling"""
    if not content.strip():
        return ""
    lines = content.split('\n')
    result = []
    for line in lines:
        if line.strip():
            result.append(f"data: {line}")
        else:
            result.append("data: ")
    return '\n'.join(result) + '\n\n'

# Chat Endpoint with Memory
@app.post("/chat")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    current_time = datetime.now().strftime("%B %d, %Y, %I:%M %p %Z")
    session_id = request.session_id or f"sess_{int(datetime.now().timestamp())}_{os.urandom(4).hex()}"

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            accumulated_response = ""
            messages = [
                {"role": "system", "content": CHAT_SYSTEM_PROMPT.format(current_time=current_time)}
            ] + load_conversation(session_id) + [
                {"role": "user", "content": request.message}
            ]

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
                        if char in [' ', '\n', '.', '!', '?', ';', ':', ','] and word_buffer.strip():
                            if '\n' in word_buffer:
                                parts = word_buffer.split('\n')
                                for i, part in enumerate(parts):
                                    if part.strip():
                                        yield f"data: {part.strip()}\n\n"
                                    if i < len(parts) - 1:
                                        yield "data: \n\n"
                            else:
                                yield f"data: {word_buffer}\n\n"
                            word_buffer = ""
            
            if word_buffer.strip():
                if '\n' in word_buffer:
                    parts = word_buffer.split('\n')
                    for i, part in enumerate(parts):
                        if part.strip():
                            yield f"data: {part.strip()}\n\n"
                        if i < len(parts) - 1:
                            yield "data: \n\n"
                else:
                    yield f"data: {word_buffer}\n\n"

            logger.info(f"Final chat response for session {session_id}:\n{accumulated_response}")

            urls = url_pattern.findall(accumulated_response)
            if urls:
                yield "data: \n\n"
                yield "data: ## Sources\n\n"
                for url in urls:
                    yield f"data: - {await check_url(url)}\n\n"

            save_message(session_id, "user", request.message)
            save_message(session_id, "assistant", accumulated_response)

            yield "data: \n\n"
            yield f"data: **Continue this chat** with session ID: `{session_id}`\n\n"

        except Exception as e:
            error_msg = f"## Error\n\n{str(e)}"
            logger.error(f"Stream error: {error_msg}")
            yield f"data: {error_msg}\n\n"
            save_message(session_id, "assistant", f"[System error: {str(e)}]")

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Web Search with Memory
@app.post("/web_search")
async def web_search(request: ChatRequest) -> StreamingResponse:
    current_time = datetime.now().strftime("%B %d, %Y, %I:%M %p %Z")
    query = request.message
    session_id = request.session_id or f"sess_{int(datetime.now().timestamp())}_{os.urandom(4).hex()}"

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            summary_prompt = WEB_SEARCH_SYSTEM_PROMPT.format(current_time=current_time) + "\n\nAnalyze and summarize the following sources:\n\n"
            all_urls = []

            try:
                dd_results = DDGS().text(query, max_results=5)
                if dd_results:
                    summary_prompt += "## Web Results (DuckDuckGo)\n"
                    for i, r in enumerate(dd_results, 1):
                        summary_prompt += f"{i}. [{r['title']}]({r['href']})\n{r['body']}\n"
                        all_urls.append(r['href'])
            except Exception as e:
                summary_prompt += f"⚠️ Could not fetch web results: {str(e)}\n"

            try:
                wiki_page = wiki.page(query)
                if wiki_page.exists():
                    summary_prompt += f"## Wikipedia Entry: {wiki_page.title}\n{wiki_page.summary[:1200]}{'...' if len(wiki_page.summary) > 1200 else ''}\n"
                    all_urls.append(wiki_page.fullurl)
                else:
                    summary_prompt += f"⚠️ No Wikipedia page found for '{query}'.\n"
            except Exception as e:
                summary_prompt += f"⚠️ Error fetching Wikipedia: {str(e)}\n"

            if "⚠️ Could not fetch" in summary_prompt and "## Wikipedia Entry" not in summary_prompt:
                response = f"## No Results Found\n\nNo reliable information found for '{query}' as of {current_time}.\n\n**Tip**: Try rephrasing your query or using more specific terms."
                yield f"data: {response}\n\n"
                save_message(session_id, "user", query)
                save_message(session_id, "assistant", response)
                yield "data: \n\n"
                yield f"data: **Continue** with session ID: `{session_id}`\n\n"
                logger.info(f"Final web search response for session {session_id}:\n{response}")
                return

            context_messages = [
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": f"Summarize the key facts about: {query}"}
            ] + load_conversation(session_id)[-4:]

            stream = client.chat.completions.create(
                model="ai/gemma3",
                messages=context_messages,
                stream=True,
                temperature=0.7,
                max_tokens=2048
            )

            accumulated_response = ""
            word_buffer = ""
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    accumulated_response += content
                    
                    for char in content:
                        word_buffer += char
                        if char in [' ', '\n', '.', '!', '?', ';', ':', ','] and word_buffer.strip():
                            if '\n' in word_buffer:
                                parts = word_buffer.split('\n')
                                for i, part in enumerate(parts):
                                    if part.strip():
                                        yield f"data: {part.strip()}\n\n"
                                    if i < len(parts) - 1:
                                        yield "data: \n\n"
                            else:
                                yield f"data: {word_buffer}\n\n"
                            word_buffer = ""
            
            if word_buffer.strip():
                if '\n' in word_buffer:
                    parts = word_buffer.split('\n')
                    for i, part in enumerate(parts):
                        if part.strip():
                            yield f"data: {part.strip()}\n\n"
                        if i < len(parts) - 1:
                            yield "data: \n\n"
                else:
                    yield f"data: {word_buffer}\n\n"

            logger.info(f"Final web search response for session {session_id}:\n{accumulated_response}")

            if all_urls:
                yield "data: \n\n"
                yield "data: ## Sources\n\n"
                for url in all_urls:
                    yield f"data: - {await check_url(url)}\n\n"

            yield "data: \n\n"
            yield "data: **Follow-Up**: Ask more questions to dive deeper!\n\n"
            yield f"data: **Continue** with session ID: `{session_id}`\n\n"

            save_message(session_id, "user", query)
            save_message(session_id, "assistant", accumulated_response)

        except Exception as e:
            error_msg = f"## Error\n\nSearch failed: {str(e)}"
            logger.error(f"Search error: {error_msg}")
            yield f"data: {error_msg}\n\n"
            save_message(session_id, "user", query)
            save_message(session_id, "assistant", f"[Error: {str(e)}]")

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Session Management Endpoints
@app.get("/sessions")
async def list_sessions():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT session_id FROM conversations ORDER BY timestamp")
        sessions = [row["session_id"] for row in cur.fetchall()]
    return {"sessions": sessions}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content, timestamp FROM conversations WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        rows = cur.fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session_id,
            "messages": [{"role": r["role"], "content": r["content"], "time": r["timestamp"]} for r in rows]
        }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM conversations WHERE session_id = ?", (session_id,))
        count = cur.fetchone()[0]
        if count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        cur.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        conn.commit()
    return {"message": f"Session {session_id} deleted successfully"}

# Health Check
@app.get("/")
async def root():
    return {"message": "NOVA Assistant is running", "endpoints": ["/chat", "/web_search", "/sessions"]}
