-- Create the checkpoints table
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id VARCHAR(255) PRIMARY KEY,
    chain_id VARCHAR(255) NOT NULL,
    chain_status VARCHAR(50) NOT NULL,
    state_class VARCHAR(255) NOT NULL,
    state_version VARCHAR(50),
    data JSONB NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    next_execution_node VARCHAR(255),
    executed_nodes JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_checkpoints_chain_id ON checkpoints(chain_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_timestamp ON checkpoints(timestamp);