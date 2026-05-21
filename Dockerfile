FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV LLM_API_KEY=""
ENV LLM_BASE_URL="https://api.openai.com/v1"
ENV LLM_MODEL="gpt-4o-mini"

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
