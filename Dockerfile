# Image Python stable compatible avec torch/vtk
FROM python:3.10-slim

WORKDIR /app

# Dépendances système nécessaires pour OpenCV, VTK, etc.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libxau6 \
    libxdmcp6 \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY server.py .

# Exposer le port pour Render
EXPOSE 5000

# Lancer le serveur avec gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "server:app"]
