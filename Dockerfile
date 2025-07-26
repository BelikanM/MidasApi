FROM python:3.10-slim  # 3.13 pas encore bien supporté pour torch/vtk

WORKDIR /app

# Installer les bibliothèques système nécessaires à VTK, Torch, Trimesh, etc.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libxau6 \
    libxdmcp6 \
    libglfw3 \
    libgles2-mesa-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "server:app"]
