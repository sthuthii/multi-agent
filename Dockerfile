FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn gradio

# Copy project
COPY . .

# Create required directories
RUN mkdir -p logs evals

EXPOSE 7860

CMD ["python", "ui/app.py"]