# Invest

Projeto de pipeline quantitativo para mercado brasileiro, com foco em:

- ingestão de dados de ações
- enriquecimento de empresas
- atualização incremental de preços
- cálculo de features
- geração de dataset para modelagem
- treino de modelo
- backtest e análise de resultados

---

# Estrutura do projeto

## Código

- `apps/ingestion/`
  - lógica de ingestão, providers e repositories
- `jobs/ingestion/`
  - pontos de execução da ingestão
- `jobs/training/`
  - pontos de execução do pipeline de treino
- `core/`
  - configuração, conexão com banco e logging
- `domain/`
  - modelos compartilhados

## Dados e artefatos

- `artifacts/datasets/`
  - datasets gerados para modelagem
- `artifacts/models/`
  - modelos treinados e artefatos auxiliares
- `artifacts/reports/`
  - métricas, predições, backtests e arquivos de análise
- `notebooks/`
  - notebooks de exploração e análise

## Infra

- `docker-compose.yml`
  - serviços principais
- `infra/docker/`
  - Dockerfiles
- `infra/postgres/init/`
  - scripts SQL de inicialização do banco
- `infra/requirements/`
  - dependências separadas por serviço

---

# Pré-requisitos

- Docker
- Docker Compose
- arquivo `.env` configurado com as variáveis do projeto

Exemplo de variáveis esperadas:

- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `database_url`

---

# Fluxo de execução

## 1. Build das imagens

```bash
make build
```

---

# Acesso ao Banco

Use variaveis do .env

```
docker compose exec db psql -U $POSTGRES_USER -d $POSTGRES_DB
```