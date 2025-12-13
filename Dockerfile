FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    duckduckgo_search \
    httpx \
    openai \
    wikipedia-api


COPY . .

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--proxy-headers", \
     "--timeout-keep-alive", "300"]

