CREATE TABLE IF NOT EXISTS submissions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    prediction SMALLINT NOT NULL,
    true_label SMALLINT NOT NULL,
    confidence FLOAT NOT NULL
);
