FROM python:3.12-slim

WORKDIR /app

# Non-root user with UID 1000 for Hugging Face Spaces and container security
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Install serving dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API serving code and exported champion model artifact
COPY --chown=user:user api/ api/
COPY --chown=user:user artifacts/ artifacts/

USER user

# Default port 7860 matches Hugging Face Spaces app_port, respects $PORT env var if provided (e.g. Render/Cloud Run)
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
