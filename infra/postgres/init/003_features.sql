CREATE TABLE IF NOT EXISTS market_data.features (
    symbol_id UUID NOT NULL REFERENCES market_data.symbols(id) ON DELETE CASCADE,
    trade_date DATE NOT NULL,

    return_1d NUMERIC(18,8),
    return_5d NUMERIC(18,8),
    return_21d NUMERIC(18,8),
    log_return_1d NUMERIC(18,8),

    sma_20 NUMERIC(18,6),
    sma_50 NUMERIC(18,6),
    price_sma_20_ratio NUMERIC(18,8),

    volatility_21d NUMERIC(18,8),

    volume_ratio_20d NUMERIC(18,8),
    avg_daily_volume_20d NUMERIC(20,2),
    avg_daily_traded_value_20d NUMERIC(20,2),

    high_low_ratio NUMERIC(18,8),
    gap NUMERIC(18,8),

    target_5d NUMERIC(18,8),
    target_5d_t1 NUMERIC(18,8),
    target_21d NUMERIC(18,8),
    target_21d_t1 NUMERIC(18,8),
    target_63d NUMERIC(18,8),
    target_63d_t1 NUMERIC(18,8),

    return_63d NUMERIC(18,8),
    return_126d NUMERIC(18,8),
    volatility_63d NUMERIC(18,8),
    volatility_ratio_21_63 NUMERIC(18,8),
    return_21d_over_vol_21d NUMERIC(18,8),
    momentum_21_63 NUMERIC(18,8),
    momentum_5_21 NUMERIC(18,8),
    volume_trend_5_20 NUMERIC(18,8),
    traded_value_trend_5_20 NUMERIC(18,8),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (symbol_id, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_features_trade_date
ON market_data.features (trade_date);

CREATE INDEX IF NOT EXISTS idx_features_symbol_date
ON market_data.features(symbol_id, trade_date DESC);

CREATE SCHEMA IF NOT EXISTS macro_data;

CREATE TABLE IF NOT EXISTS macro_data.market_features (
    trade_date DATE PRIMARY KEY,

    -- =========================
    -- IBOV
    -- =========================
    ibov_close DOUBLE PRECISION,
    ibov_return_1d DOUBLE PRECISION,
    ibov_return_5d DOUBLE PRECISION,
    ibov_return_21d DOUBLE PRECISION,
    ibov_return_63d DOUBLE PRECISION,
    ibov_vol_21d DOUBLE PRECISION,
    ibov_vol_63d DOUBLE PRECISION,
    ibov_above_sma_200 DOUBLE PRECISION,

    -- =========================
    -- S&P 500
    -- =========================
    sp500_close DOUBLE PRECISION,
    sp500_return_1d DOUBLE PRECISION,
    sp500_return_5d DOUBLE PRECISION,
    sp500_return_21d DOUBLE PRECISION,
    sp500_return_63d DOUBLE PRECISION,
    sp500_vol_21d DOUBLE PRECISION,
    sp500_vol_63d DOUBLE PRECISION,
    sp500_above_sma_200 DOUBLE PRECISION,

    -- =========================
    -- SELIC
    -- =========================
    selic_rate DOUBLE PRECISION,
    selic_change_21d DOUBLE PRECISION,
    selic_change_63d DOUBLE PRECISION,

    -- =========================
    -- IPCA
    -- =========================
    ipca_monthly DOUBLE PRECISION,
    ipca_3m DOUBLE PRECISION,
    ipca_6m DOUBLE PRECISION,
    ipca_12m DOUBLE PRECISION,
    ipca_change_3m DOUBLE PRECISION,
    ipca_change_6m DOUBLE PRECISION,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_market_features_trade_date
ON macro_data.market_features (trade_date);