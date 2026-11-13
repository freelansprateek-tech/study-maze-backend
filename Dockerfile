# Study Maze backend: container for Hugging Face Spaces (Docker SDK, CPU).
# Spaces runs the container as user 1000 and expects the app on port 7860.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /home/user/app

# CPU-only PyTorch first: the default wheel bundles GPU libraries the Space cannot use.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake both models into the image so a waking Space does not download them again.
RUN python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-1.5B-Instruct'); d('BAAI/bge-small-en-v1.5')"

COPY --chown=user . .

# Defaults for the hosted copy. LLM_API_KEY is set as a Space secret, never here.
ENV LLM_PROVIDER=openai \
    FRONTEND_ORIGIN=https://study-maze-frontend.vercel.app \
    DEVICE=cpu

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
