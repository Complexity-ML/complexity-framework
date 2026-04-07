FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir aiohttp

COPY app.py /app/app.py

EXPOSE 7860

CMD ["python", "app.py"]
