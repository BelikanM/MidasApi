FROM python:3.13-slim

WORKDIR /app

# 📦 Installe dépendances système nécessaires
RUN apt-get update && apt-get install -y libx11-6 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "server:app"]
