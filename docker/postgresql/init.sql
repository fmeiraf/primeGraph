-- Create the checkpoints table
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id VARCHAR PRIMARY KEY,
    chain_id VARCHAR NOT NULL,
    chain_status INTEGER NOT NULL,
    state_class VARCHAR NOT NULL,
    state_version VARCHAR,
    data JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    next_execution_node VARCHAR,
    executed_nodes JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_checkpoints_chain_id ON checkpoints(chain_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_timestamp ON checkpoints(timestamp);