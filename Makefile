.PHONY: help build build-ingestion build-training up-db down-db logs-db \
        ingest-full ingest-update ingest-bootstrap ingest-enrich ingest-prices ingest-features \
        train-full train-dataset train-model train-backtest train-analyze train-validation \
        ps clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make build               - Builda ingestion e training"
	@echo "  make build-ingestion     - Builda só a imagem de ingestion"
	@echo "  make build-training      - Builda só a imagem de training"
	@echo "  make up-db               - Sobe apenas o banco"
	@echo "  make down-db             - Derruba os containers"
	@echo "  make reset-db            - Faz reset completo do banco"
	@echo "  make delete-db           - Deleta o banco"
	@echo "  make logs-db             - Mostra logs do banco"
	@echo "  make ps                  - Mostra status dos serviços"
	@echo ""
	@echo "Ingestion:"
	@echo "  make ingest-full         - Roda pipeline completa de ingestão"
	@echo "  make ingest-update       - Roda atualização incremental"
	@echo "  make ingest-bootstrap    - Carrega universo de símbolos"
	@echo "  make ingest-enrich       - Enriquece empresas/fundamentos"
	@echo "  make ingest-prices       - Ingere preços diários"
	@echo "  make ingest-market       - Ingere features de mercado"
	@echo "  make ingest-features     - Calcula features"
	@echo ""
	@echo "Training:"
	@echo "  make train-full          - Roda pipeline completa de treino"
	@echo "  make train-dataset       - Gera dataset de modelagem"
	@echo "  make train-model         - Treina o modelo"
	@echo "  make train-backtest      - Roda backtest"
	@echo "  make train-analyze       - Roda análise das predições"
	@echo "  make train-feature-ic    - Analisa relação das features com os valores de IC"
	@echo "  make train-experiment    - Testa grids de valores em treino e backtest"
	@echo "  make train-walk    	  - Roda backtest walkforward"
	@echo ""
	@echo "Outros:"
	@echo "  make clean               - Remove containers parados"
	@echo "  make reset-features      - Remove as features do banco de dados"

build:
	docker compose build ingestion training

build-ingestion:
	docker compose build ingestion

build-training:
	docker compose build training

up-db:
	docker compose up -d db

down-db:
	docker compose down

logs-db:
	docker compose logs db

ps:
	docker compose ps

ingest-full:
	docker compose run --rm ingestion python -m jobs.ingestion.full_build

ingest-update:
	docker compose run --rm ingestion python -m jobs.ingestion.incremental_update

ingest-bootstrap:
	docker compose run --rm ingestion python -m jobs.ingestion.bootstrap_universe

ingest-enrich:
	docker compose run --rm ingestion python -m jobs.ingestion.enrich_companies

ingest-prices:
	docker compose run --rm ingestion python -m jobs.ingestion.ingest_daily_prices

ingest-market:
	docker compose run --rm ingestion python -m jobs.ingestion.market

ingest-features:
	docker compose run --rm ingestion python -m jobs.ingestion.compute_features

train-full:
	docker compose run --rm training python -m jobs.training.full_train

train-dataset:
	docker compose run --rm training python -m jobs.training.build_dataset

train-model:
	docker compose run --rm training python -m jobs.training.train_model

train-backtest:
	docker compose run --rm training python -m jobs.training.backtest

train-analyze:
	docker compose run --rm training python -m jobs.training.analyze

train-validation:
	docker compose run --rm training python -m jobs.training.validation

train-feature-ic:
	docker compose run --rm training python -m jobs.training.analyze_feature_ic

make train-experiment:
	docker compose run --rm training python -m jobs.training.run_experiment_grid

make train-walk:
	docker compose run --rm training python -m jobs.training.walkforward

clean:
	docker container prune -f

reset-features:
	docker compose exec db psql -U stocks_user -d stocks -c "TRUNCATE TABLE market_data.features CASCADE;"

reset-db:
	docker compose down -v
	docker compose up -d db

delete-db:
	docker compose down -v