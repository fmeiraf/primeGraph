import logging
from datetime import datetime
from typing import List, Optional, Set

from psycopg2.extras import DictCursor, Json
from psycopg2.pool import SimpleConnectionPool

from tiny_graph.checkpoint.base import StorageBackend
from tiny_graph.checkpoint.serialization import serialize_model
from tiny_graph.graph.executable import ChainStatus
from tiny_graph.models.checkpoint import Checkpoint
from tiny_graph.models.state import GraphState

logger = logging.getLogger(__name__)

CREATE_TABLES_SQL = """
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

CREATE INDEX IF NOT EXISTS idx_checkpoints_chain_id ON checkpoints(chain_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_timestamp ON checkpoints(timestamp);
"""


class PostgreSQLStorage(StorageBackend):
    def __init__(
        self,
        dsn: str,
        min_connections: int = 1,
        max_connections: int = 10,
    ):
        """Initialize PostgreSQL storage backend.

        Args:
            dsn: Database connection string
            min_connections: Minimum number of connections in pool
            max_connections: Maximum number of connections in pool
        """
        super().__init__()
        self.dsn = dsn
        self.pool = SimpleConnectionPool(
            minconn=min_connections, maxconn=max_connections, dsn=dsn
        )

    def initialize_database(self):
        """Create necessary database tables and indexes."""
        with self.pool.getconn() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(CREATE_TABLES_SQL)
                conn.commit()
                logger.info("Database tables initialized successfully")
            finally:
                self.pool.putconn(conn)

    def save_checkpoint(
        self,
        state_instance: GraphState,
        chain_id: str,
        chain_status: ChainStatus,
        checkpoint_id: Optional[str] = None,
        next_execution_node: Optional[str] = None,
        executed_nodes: Optional[Set[str]] = None,
    ) -> str:
        checkpoint_id = self._enforce_checkpoint_id(checkpoint_id)
        self._enforce_same_model_version(state_instance, chain_id)

        state_class_str = (
            f"{state_instance.__class__.__module__}.{state_instance.__class__.__name__}"
        )

        serialized_data = serialize_model(state_instance)

        sql = """
        INSERT INTO checkpoints (
            checkpoint_id, chain_id, chain_status, state_class, 
            state_version, data, timestamp, next_execution_node, executed_nodes
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (checkpoint_id) 
        DO UPDATE SET
            chain_status = EXCLUDED.chain_status,
            data = EXCLUDED.data,
            timestamp = EXCLUDED.timestamp,
            next_execution_node = EXCLUDED.next_execution_node,
            executed_nodes = EXCLUDED.executed_nodes
        """

        with self.pool.getconn() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        sql,
                        (
                            checkpoint_id,
                            chain_id,
                            chain_status.value,
                            state_class_str,
                            getattr(state_instance, "version", None),
                            Json(serialized_data),
                            datetime.now(),
                            next_execution_node,
                            Json(list(executed_nodes)) if executed_nodes else None,
                        ),
                    )
                conn.commit()
                logger.info(f"Checkpoint '{checkpoint_id}' saved to PostgreSQL")
                return checkpoint_id
            finally:
                self.pool.putconn(conn)

    def load_checkpoint(
        self, state_instance: GraphState, chain_id: str, checkpoint_id: str
    ) -> Checkpoint:
        self._enforce_same_model_version(state_instance, chain_id)

        sql = """
        SELECT * FROM checkpoints 
        WHERE chain_id = %s AND checkpoint_id = %s
        """

        with self.pool.getconn() as conn:
            try:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(sql, (chain_id, checkpoint_id))
                    result = cur.fetchone()

                    if not result:
                        raise KeyError(
                            f"Checkpoint '{checkpoint_id}' not found for chain '{chain_id}'"
                        )

                    return Checkpoint(
                        checkpoint_id=result["checkpoint_id"],
                        chain_id=result["chain_id"],
                        chain_status=ChainStatus(result["chain_status"]),
                        state_class=result["state_class"],
                        state_version=result["state_version"],
                        data=result["data"],
                        timestamp=result["timestamp"],
                        next_execution_node=result["next_execution_node"],
                        executed_nodes=set(result["executed_nodes"])
                        if result["executed_nodes"]
                        else None,
                    )
            finally:
                self.pool.putconn(conn)

    def list_checkpoints(self, chain_id: str) -> List[Checkpoint]:
        sql = """
        SELECT * FROM checkpoints 
        WHERE chain_id = %s 
        ORDER BY timestamp DESC
        """

        with self.pool.getconn() as conn:
            try:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(sql, (chain_id,))
                    results = cur.fetchall()

                    return [
                        Checkpoint(
                            checkpoint_id=row["checkpoint_id"],
                            chain_id=row["chain_id"],
                            chain_status=ChainStatus(row["chain_status"]),
                            state_class=row["state_class"],
                            state_version=row["state_version"],
                            data=row["data"],
                            timestamp=row["timestamp"],
                            next_execution_node=row["next_execution_node"],
                            executed_nodes=set(row["executed_nodes"])
                            if row["executed_nodes"]
                            else None,
                        )
                        for row in results
                    ]
            finally:
                self.pool.putconn(conn)

    def delete_checkpoint(self, checkpoint_id: str, chain_id: str) -> None:
        sql = """
        DELETE FROM checkpoints 
        WHERE chain_id = %s AND checkpoint_id = %s
        RETURNING checkpoint_id
        """

        with self.pool.getconn() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (chain_id, checkpoint_id))
                    if cur.rowcount == 0:
                        raise KeyError(
                            f"Checkpoint '{checkpoint_id}' not found for chain '{chain_id}'"
                        )
                conn.commit()
                logger.info(f"Checkpoint '{checkpoint_id}' deleted from PostgreSQL")
            finally:
                self.pool.putconn(conn)

    def get_last_checkpoint_id(self, chain_id: str) -> Optional[str]:
        sql = """
        SELECT checkpoint_id 
        FROM checkpoints 
        WHERE chain_id = %s 
        ORDER BY timestamp DESC 
        LIMIT 1
        """

        with self.pool.getconn() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, (chain_id,))
                    result = cur.fetchone()
                    return result[0] if result else None
            finally:
                self.pool.putconn(conn)

    def __del__(self):
        """Cleanup connection pool on object destruction."""
        if hasattr(self, "pool"):
            self.pool.closeall()
