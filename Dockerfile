# Stage 1: Build Frontend UI
FROM node:22-alpine AS frontend-builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Stage 2: Backend API + Background Worker + Static Frontend
FROM python:3.12-slim
WORKDIR /srv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY backend/requirements.txt ./
RUN pip install -r requirements.txt

COPY backend/ ./
COPY --from=frontend-builder /app/dist /srv/static

RUN chmod +x start.sh

EXPOSE 8000

CMD ["bash", "start.sh"]
