FROM python:3.11-slim
WORKDIR /app
COPY gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY gateway /app/gateway
COPY shared /app/shared
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["python", "gateway/main.py"]