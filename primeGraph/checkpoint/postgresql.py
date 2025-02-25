import enum
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Literal, Optional

import psycopg2
from psycopg2 import DatabaseError, OperationalError
from psycopg2.extras import DictCursor, Json
from psycopg2.pool import ThreadedConnectionPool
from pydantic import BaseModel

from primeGraph.checkpoint.base import CheckpointData, StorageBackend
from primeGraph.checkpoint.serialization import serialize_model
from primeGraph.graph.engine import ExecutionFrame
from primeGraph.models.checkpoint import Checkpoint
from primeGraph.models.state import GraphState
from primeGraph.types import ChainStatus

logger = logging.getLogger(__name__)


@dataclass
class PostgreSQLConfig:
    dsn: str
    min_connections: int = 1
    max_connections: int = 10
    connection_timeout: int = 30
    retry_attempts: int = 3
    isolation_level: Literal["serializable", "repeatable read", "read committed", "read uncommitted"] = "read committed"


class PostgreSQLStorage(StorageBackend):
    def __init__(self, config: PostgreSQLConfig):
        """Initialize PostgreSQL storage backend with enhanced configuration."""
        super().__init__()
        self.dsn = config.dsn
        self.retry_attempts = config.retry_attempts
        self.isolation_level = config.isolation_level
        self.connection_timeout = config.connection_timeout

        # Use connection factory with timeout and health checks
        self.pool = ThreadedConnectionPool(
            minconn=config.min_connections,
            maxconn=config.max_connections,
            dsn=config.dsn,
            connection_factory=self._create_connection_with_timeout,
        )

    def _create_connection_with_timeout(self, dsn: Optional[str] = None) -> psycopg2.extensions.connection:
        """Create connection with timeout and proper isolation level.

        Args:
            dsn: Database connection string. If None, uses self.dsn
        """
        dsn = dsn or self.dsn
        # First create the connection without isolation level
        conn = psycopg2.connect(
            dsn,
            connect_timeout=self.connection_timeout,
        )

        # Set isolation level after connection is established
        if self.isolation_level == "read committed":
            conn.isolation_level = psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED
        elif self.isolation_level == "read uncommitted":
            conn.isolation_level = psycopg2.extensions.ISOLATION_LEVEL_READ_UNCOMMITTED
        elif self.isolation_level == "repeatable read":
            conn.isolation_level = psycopg2.extensions.ISOLATION_LEVEL_REPEATABLE_READ
        elif self.isolation_level == "serializable":
            conn.isolation_level = psycopg2.extensions.ISOLATION_LEVEL_SERIALIZABLE

        return conn

    def _convert_sets_to_lists(self, obj: Any) -> Any:
        """Helper method to convert sets to lists in nested structures."""
        # Specialized handling for GraphState and ExecutionFrame must come first
        if isinstance(obj, GraphState):
            return {"__class__": f"{obj.__class__.__module__}.{obj.__class__.__name__}", "data": serialize_model(obj)}
        elif isinstance(obj, ExecutionFrame):
            return {
                "__class__": "primeGraph.graph.engine.ExecutionFrame",
                "data": {
                    "node_id": obj.node_id,
                    "state": self._convert_sets_to_lists(obj.state),
                    "branch_id": obj.branch_id,
                    "target_convergence": obj.target_convergence,
                    "resumed": obj.resumed,
                },
            }

        if isinstance(obj, BaseModel):
            return self._convert_sets_to_lists(obj.model_dump())

        if isinstance(obj, enum.Enum):
            return obj.value

        # Use model_dump if available (pydantic v2), else try dict() (pydantic v1)
        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            return self._convert_sets_to_lists(obj.model_dump())
        elif hasattr(obj, "dict") and callable(obj.dict):
            return self._convert_sets_to_lists(obj.dict())
        elif hasattr(obj, "__dict__") and not isinstance(obj, dict):
            return self._convert_sets_to_lists(vars(obj))

        if isinstance(obj, dict):
            return {key: self._convert_sets_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, list):
            return [self._convert_sets_to_lists(item) for item in obj]

        # Fallback: if the object isn't one of the basic serializable types, convert it to string
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def _convert_lists_to_sets(self, obj: Any) -> Any:
        """Helper method to convert lists back to sets in nested structures.
        Only converts lists that were originally sets based on context."""
        if isinstance(obj, dict):
            # Check if this is a serialized GraphState or ExecutionFrame
            if "__class__" in obj and "data" in obj:
                class_path = obj["__class__"]
                if class_path == "primeGraph.graph.engine.ExecutionFrame":
                    # Reconstruct ExecutionFrame from serialized dict
                    data_field = obj["data"]
                    frame_data = self._convert_lists_to_sets(data_field)
                    if isinstance(frame_data, dict):
                        frame = ExecutionFrame(node_id=frame_data["node_id"], state=frame_data["state"])
                        frame.branch_id = frame_data["branch_id"]
                        frame.target_convergence = frame_data["target_convergence"]
                        frame.resumed = frame_data["resumed"]
                        return frame
                    elif isinstance(frame_data, ExecutionFrame):
                        return frame_data
                    else:
                        return frame_data
                else:
                    # Handle GraphState as before
                    module_name, class_name = class_path.rsplit(".", 1)
                    module = __import__(module_name, fromlist=[class_name])
                    state_class = getattr(module, class_name)
                    if isinstance(obj["data"], str):
                        import json

                        data = json.loads(obj["data"])
                    else:
                        data = obj["data"]
                    return state_class(**data)

            # Additional check: if the dict looks like an ExecutionFrame without the __class__ marker
            required_keys = ["node_id", "state", "branch_id", "target_convergence", "resumed"]
            if all(key in obj for key in required_keys):
                frame = ExecutionFrame(node_id=obj["node_id"], state=obj["state"])
                frame.branch_id = obj["branch_id"]
                frame.target_convergence = obj["target_convergence"]
                frame.resumed = obj["resumed"]
                return frame

            # Special handling for known set fields
            if "visited_nodes" in obj:
                obj["visited_nodes"] = set(obj["visited_nodes"])
            if "active_branches" in obj:
                obj["active_branches"] = {
                    k: set(v) if isinstance(v, list) else v for k, v in obj["active_branches"].items()
                }
            return {key: self._convert_lists_to_sets(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_lists_to_sets(item) for item in obj]
        return obj

    def save_checkpoint(
        self,
        state_instance: GraphState,
        checkpoint_data: CheckpointData,
    ) -> str:
        checkpoint_id = self._enforce_checkpoint_id(checkpoint_data.checkpoint_id)
        self._enforce_same_model_version(state_instance, checkpoint_data.chain_id)

        state_class_str = f"{state_instance.__class__.__module__}.{state_instance.__class__.__name__}"
        serialized_data = serialize_model(state_instance)

        sql = """
        INSERT INTO checkpoints (
            checkpoint_id, chain_id, chain_status, state_class,
            state_version, data, timestamp, engine_state
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (checkpoint_id)
        DO UPDATE SET
            chain_status = EXCLUDED.chain_status,
            data = EXCLUDED.data,
            timestamp = EXCLUDED.timestamp,
            engine_state = EXCLUDED.engine_state
        """

        for attempt in range(self.retry_attempts):
            conn = self.pool.getconn()
            try:
                with conn.cursor() as cur:
                    # Add advisory lock to prevent concurrent updates
                    cur.execute("SELECT pg_advisory_xact_lock(%s)", (hash(checkpoint_id),))

                    # Convert engine_state to JSON format, ensuring sets are converted to lists
                    engine_state = (
                        self._convert_sets_to_lists(checkpoint_data.engine_state)
                        if checkpoint_data.engine_state
                        else None
                    )
                    engine_state_json = Json(engine_state) if engine_state else None

                    cur.execute(
                        sql,
                        (
                            checkpoint_id,
                            checkpoint_data.chain_id,
                            checkpoint_data.chain_status.value,
                            state_class_str,
                            getattr(state_instance, "version", None),
                            Json(serialized_data),
                            datetime.now(),
                            engine_state_json,
                        ),
                    )
                    conn.commit()
                    logger.info(f"Checkpoint '{checkpoint_id}' saved to PostgreSQL")
                    return checkpoint_id
            except (OperationalError, DatabaseError):
                if attempt == self.retry_attempts - 1:
                    raise
                time.sleep(0.1 * (2**attempt))  # Exponential backoff
            finally:
                self.pool.putconn(conn)

        raise RuntimeError(f"Failed to save checkpoint after {self.retry_attempts} attempts")

    def load_checkpoint(self, state_instance: GraphState, chain_id: str, checkpoint_id: str) -> Checkpoint:
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
                        raise KeyError(f"Checkpoint '{checkpoint_id}' not found for chain '{chain_id}'")

                    # Convert lists back to sets in engine_state if it exists
                    engine_state = result["engine_state"]
                    if engine_state:
                        engine_state = self._convert_lists_to_sets(engine_state)

                    return Checkpoint(
                        checkpoint_id=result["checkpoint_id"],
                        chain_id=result["chain_id"],
                        chain_status=ChainStatus(result["chain_status"]),
                        state_class=result["state_class"],
                        state_version=result["state_version"],
                        data=result["data"],
                        timestamp=result["timestamp"],
                        engine_state=engine_state,
                    )
            finally:
                self.pool.putconn(conn)

    def list_checkpoints(self, chain_id: str) -> List[Checkpoint]:
        sql = """
        SELECT * FROM checkpoints
        WHERE chain_id = %s
        ORDER BY timestamp ASC
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
                            engine_state=row["engine_state"],
                        )
                        for row in results
                    ]
            finally:
                self.pool.putconn(conn)

    def delete_checkpoint(self, chain_id: str, checkpoint_id: str) -> None:
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
                        raise KeyError(f"Checkpoint '{checkpoint_id}' not found for chain '{chain_id}'")
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

    def __del__(self) -> None:
        """Cleanup connection pool on object destruction."""
        if hasattr(self, "pool"):
            self.pool.closeall()

    def check_schema(self) -> bool:
        """Check if the required tables and columns exist in the database.

        Returns:
            bool: True if schema is valid, False otherwise
        """
        check_table_sql = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'checkpoints'
        );
        """

        check_columns_sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'checkpoints';
        """

        required_columns = {
            "checkpoint_id",
            "chain_id",
            "chain_status",
            "state_class",
            "state_version",
            "data",
            "timestamp",
            "engine_state",
            "created_at",
        }

        with self.pool.getconn() as conn:
            try:
                with conn.cursor() as cur:
                    # Check if table exists
                    cur.execute(check_table_sql)
                    table_exists = cur.fetchone()[0]

                    if not table_exists:
                        logger.warning("Checkpoints table does not exist")
                        return False

                    # Check columns
                    cur.execute(check_columns_sql)
                    existing_columns = {row[0] for row in cur.fetchall()}

                    missing_columns = required_columns - existing_columns
                    if missing_columns:
                        logger.warning(f"Missing required columns: {missing_columns}")
                        return False

                    return True
            finally:
                self.pool.putconn(conn)

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "PostgreSQLStorage":
        """Create a PostgreSQLStorage instance from a database URL.

        Args:
            url: Database URL in format:
                postgresql://user:password@host:port/dbname
            **kwargs: Additional connection pool parameters

        Returns:
            PostgreSQLStorage: Configured storage instance
        """
        return cls(config=PostgreSQLConfig(dsn=url, **kwargs))

    @classmethod
    def from_config(
        cls,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 5432,
        **kwargs: Any,
    ) -> "PostgreSQLStorage":
        """Create a PostgreSQLStorage instance from individual configuration parameters.

        Args:
            host: Database host
            database: Database name
            user: Username
            password: Password
            port: Database port (default: 5432)
            **kwargs: Additional connection pool parameters

        Returns:
            PostgreSQLStorage: Configured storage instance
        """
        dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        return cls(config=PostgreSQLConfig(dsn=dsn, **kwargs))
