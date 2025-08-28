FROM ghcr.io/ggerganov/llama.cpp:server

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir duckduckgo_search httpx fastapi uvicorn openai wikipedia-api
# copy your Gemma model into the image
COPY ./models/gemma3.gguf /models/gemma3.gguf

CMD ["--model", "/models/gemma3.gguf", "--host", "0.0.0.0", "--port", "8080"]

