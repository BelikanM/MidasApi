FROM python:3.13-slim

WORKDIR /app

# ✅ Installe les bibliothèques nécessaires à VTK / OpenGL / X11
RUN apt-get update && apt-get install -y \
    libgl1 \
    libx11-6 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libxau6 \
    libxdmcp6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "server:app"]
