# Stock Data Pipeline (Brazilian Market)

## Overview

This project builds a data pipeline to collect, store, and prepare Brazilian stock market data for future analysis and machine learning models.

The current system focuses on:

* Collecting **all available Brazilian stock symbols**
* Storing **historical price data (OHLCV)**
* Enriching companies with **basic profile information**
* Structuring everything in a **relational PostgreSQL database**
* Running everything inside a **Docker environment**

---

## Current Architecture

### Components

* **Data ingestion layer** (Python)
* **PostgreSQL database**
* **Docker / Docker Compose environment**

---

## Database Schema

All data is stored under the schema:

```
market_data
```

### Tables

#### 1. `companies`

Stores company-level information.

Fields:

* `id`
* `company_name`
* `trading_name`
* `sector`
* `subsector`
* `segment`
* `description`
* `website`
* `cnpj`
* timestamps

**Current state:**

* ~289 companies enriched with:

  * sector
  * industry
  * description
  * website
* Some duplication exists (same company across multiple symbols)

---

#### 2. `symbols`

Represents tradable tickers (B3).

Fields:

* `id`
* `company_id`
* `symbol`
* `asset_type`
* `exchange`
* `currency`
* `is_active`
* timestamps

**Current state:**

* 379 active Brazilian stock symbols

---

#### 3. `daily_prices`

Stores historical price data (OHLCV).

Fields:

* `symbol_id`
* `trade_date`
* `open_price`
* `high_price`
* `low_price`
* `close_price`
* `adjusted_close_price`
* `volume`
* `source`
* `ingested_at`

**Current state:**

* ~446,000 rows
* Date range:

  * from ~2021
  * to current day
* Incremental ingestion enabled
* No duplicate rows (protected by `(symbol_id, trade_date)` constraint)

---

#### 4. `fundamentals_snapshot`

Stores fundamental data snapshots per day.

Fields include:

* `market_cap`
* `price_to_earnings`
* `dividend_yield`
* `roe`
* `roa`
* `debt_to_equity`
* etc.

**Current state:**

* Basic fields populated
* Advanced fields mostly empty due to API limitations

---

#### 5. `corporate_actions`

Currently unused (reserved for future use).

---

#### 6. `ingestion_runs`

Reserved for tracking pipeline executions (not yet fully implemented).

---

## Data Providers

### 1. Yahoo Finance (via yfinance)

Used for:

* Historical price data

Advantages:

* Free
* Reliable
* Good historical coverage

---

### 2. Brapi

Used for:

* Company profile data

Available (free plan):

* sector
* industry
* description
* website

Unavailable (free plan):

* financialData
* defaultKeyStatistics

Fallback mechanism implemented:

* tries full profile
* falls back to summaryProfile

---

## Pipeline Execution

All commands are executed via Docker.

---

### 1. Build containers

```
docker compose build ingestion
```

---

### 2. Bootstrap symbol universe

```
docker compose run --rm ingestion python scripts/bootstrap_universe.py
```

What it does:

* Fetches all Brazilian stock symbols
* Creates entries in:

  * `companies`
  * `symbols`

---

### 3. Enrich companies

```
docker compose run --rm ingestion python scripts/enrich_companies.py
```

What it does:

* Fetches company profile from brapi
* Populates:

  * sector
  * subsector
  * description
  * website
* Updates `companies` table
* Stores basic fundamentals snapshot

---

### 4. Ingest daily prices

```
docker compose run --rm ingestion python scripts/ingest_daily_prices.py
```

What it does:

* Downloads OHLCV data from Yahoo
* Stores in `daily_prices`
* Uses **incremental ingestion**:

  * loads full history on first run
  * updates only recent days afterward

---

## Key Features Implemented

### Incremental Ingestion

* Avoids re-downloading full history
* Uses last available date per symbol
* Includes overlap window for safety

---

### Idempotent Writes

* Uses `ON CONFLICT`
* No duplicate data
* Safe to rerun pipelines

---

### Multi-provider Strategy

* Yahoo → prices
* Brapi → company profile
* Fallback logic for API limitations

---

### Data Integrity Checks

Validated:

* No missing symbols in price data
* No duplicate price rows
* Consistent date coverage

---

## Known Limitations

* Companies duplicated across multiple symbols
* Limited fundamentals (due to brapi plan)
* No corporate actions integration yet
* No feature engineering yet

---

## Next Step: Feature Engineering

The next stage of the project is to transform raw price data into **model-ready features**.

---

### Objective

Create a new dataset with:

* technical indicators
* statistical features
* target variables for ML models

---

### Planned Features

#### Returns

* daily return
* log return
* 5-day return
* 21-day return

---

#### Trend

* SMA (20, 50)
* price / SMA ratio

---

#### Volatility

* rolling standard deviation (21 days)

---

#### Volume

* volume / rolling average volume

---

#### Price Structure

* (high - low) / close
* gap (open vs previous close)

---

#### Targets

* future return (e.g. 5-day ahead)
* classification labels

---

### Planned Implementation

New components:

```
services/compute_features.py
scripts/compute_features.py
```

New table:

```
market_data.features
```

Pipeline:

1. read `daily_prices`
2. compute features using pandas
3. store results in DB
4. incremental updates

---

## Project Status

Current stage:

```
Data ingestion → COMPLETE
Data enrichment → PARTIAL
Feature engineering → NEXT STEP
```

---

## Future Improvements

* Yahoo-based fundamentals integration
* Company deduplication model
* Corporate actions adjustment
* Parallel ingestion
* Feature store optimization
* ML model training

---

## Summary

The project currently provides:

* Full Brazilian stock universe
* Historical OHLCV data
* Basic company metadata
* Clean relational structure
* Incremental, production-ready ingestion pipeline

Next step is to build a **feature layer** on top of this data to enable modeling and analysis.
