# Engram memory API. The LLM (:5535) and embedder (:5502) are remote, called via stdlib in serve.py.
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ".[server,postgres]"
ENV ENGRAM_DATA_DIR=/data/engram ENGRAM_OPEN=1
EXPOSE 8000
CMD ["uvicorn", "deploy.serve:app", "--host", "0.0.0.0", "--port", "8000"]
