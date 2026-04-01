CREATE SCHEMA IF NOT EXISTS market_data;

CREATE TABLE IF NOT EXISTS market_data.companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cnpj VARCHAR(20),
    company_name TEXT NOT NULL UNIQUE,
    trading_name TEXT,
    sector TEXT,
    subsector TEXT,
    segment TEXT,
    description TEXT,
    website TEXT,
    country TEXT DEFAULT 'BR',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS market_data.symbols (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES market_data.companies(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    asset_type VARCHAR(30) NOT NULL DEFAULT 'stock',
    exchange VARCHAR(20) NOT NULL DEFAULT 'B3',
    currency VARCHAR(10) NOT NULL DEFAULT 'BRL',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS market_data.daily_prices (
    symbol_id UUID NOT NULL REFERENCES market_data.symbols(id) ON DELETE CASCADE,
    trade_date DATE NOT NULL,
    open_price NUMERIC(18,6),
    high_price NUMERIC(18,6),
    low_price NUMERIC(18,6),
    close_price NUMERIC(18,6),
    adjusted_close_price NUMERIC(18,6),
    volume BIGINT,
    trades BIGINT,
    vwap NUMERIC(18,6),
    source TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, trade_date)
);

CREATE TABLE IF NOT EXISTS market_data.fundamentals_snapshot (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol_id UUID NOT NULL REFERENCES market_data.symbols(id) ON DELETE CASCADE,
    reference_date DATE NOT NULL,
    market_cap NUMERIC(20,2),
    enterprise_value NUMERIC(20,2),
    shares_outstanding NUMERIC(20,2),
    price_to_earnings NUMERIC(20,6),
    price_to_book NUMERIC(20,6),
    eps NUMERIC(20,6),
    roe NUMERIC(20,6),
    roa NUMERIC(20,6),
    debt_to_equity NUMERIC(20,6),
    dividend_yield NUMERIC(20,6),
    beta NUMERIC(20,6),
    fifty_two_week_high NUMERIC(18,6),
    fifty_two_week_low NUMERIC(18,6),
    source TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (symbol_id, reference_date)
);

CREATE TABLE IF NOT EXISTS market_data.corporate_actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol_id UUID NOT NULL REFERENCES market_data.symbols(id) ON DELETE CASCADE,
    action_type VARCHAR(50) NOT NULL,
    approved_on DATE,
    last_date_prior DATE,
    payment_date DATE,
    value_per_share NUMERIC(18,8),
    details JSONB,
    source TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS market_data.ingestion_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    details JSONB
);

CREATE INDEX IF NOT EXISTS idx_symbols_company_id ON market_data.symbols(company_id);
CREATE INDEX IF NOT EXISTS idx_daily_prices_trade_date ON market_data.daily_prices(trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol_date ON market_data.fundamentals_snapshot(symbol_id, reference_date DESC);
CREATE INDEX IF NOT EXISTS idx_actions_symbol ON market_data.corporate_actions(symbol_id);

SELECT create_hypertable(
    'market_data.daily_prices',
    'trade_date',
    if_not_exists => TRUE,
    migrate_data => TRUE
);