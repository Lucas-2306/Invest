# Brazilian Stocks Data Platform

Projeto pessoal para ingestão, armazenamento e futura análise de ações brasileiras.

## Objetivo inicial
- Capturar o maior universo possível de ações brasileiras
- Armazenar metadados das empresas
- Armazenar séries diárias de preços
- Manter arquitetura preparada para cloud

## Stack
- Python 3.12
- PostgreSQL + TimescaleDB
- Docker / Docker Compose

## Subir ambiente
```bash
cp .env.example .env
docker compose up --build