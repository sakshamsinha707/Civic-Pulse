FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Cloud Run / Render inject PORT at runtime; default 8080
ENV PORT=8080

EXPOSE 8080

CMD ["python", "api_server.py"]
