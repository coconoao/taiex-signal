FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["python", "app.py", "--host", "0.0.0.0"]
