FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY state_manager.py .
COPY semaphore.py .
COPY rag_engine.py .
COPY adk_agents.py .
COPY mcp_server.py .
COPY config_loader.py .
# Only the app config — NEVER the service account JSON (Cloud Run uses the
# attached runtime service account via ADC, not a key file in the image).
COPY config.json .
COPY static/ static/

ENV PORT=8080
ENV GOOGLE_GENAI_USE_VERTEXAI=TRUE

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
