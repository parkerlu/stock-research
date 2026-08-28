FROM python:3.11-slim

WORKDIR /app

# Install system deps for psycopg2
RUN sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# 阿里云 PyPI 镜像 — 官方源在国内 ECS 上只有 ~58KB/s
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
# torch (CPU only) for LSTM forecast inference — 阿里云 pytorch-wheels 镜像取 +cpu 版,
# 依赖包回落到阿里云 PyPI (决不能从默认 PyPI 装 torch: linux 下会拉 CUDA 全家桶 ~2.5GB)
RUN pip install --no-cache-dir --timeout 300 torch \
    -f https://mirrors.aliyun.com/pytorch-wheels/cpu/ \
    -i https://mirrors.aliyun.com/pypi/simple/

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
