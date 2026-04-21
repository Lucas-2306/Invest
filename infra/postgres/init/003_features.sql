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

    high_low_ratio NUMERIC(18,8),
    gap NUMERIC(18,8),

    target_5d NUMERIC(18,8),
    target_5d_t1 NUMERIC(18,8),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (symbol_id, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_features_symbol_date
ON market_data.features(symbol_id, trade_date DESC);