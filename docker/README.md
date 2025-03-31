## PostgreSQL set up

This is a simple set up for PostgreSQL, mostly for local development but not meant for production.

### How the set up works

The set up is done using docker compose. The `docker-compose.yml` file contains the configuration for the PostgreSQL container and pgAdmin.

On initialization, the PostgreSQL container will run the `init.sql` file, which creates the `checkpoints` table and the necessary indexes.

### Running the container

```bash
docker compose up
```

### Accessing the database

```bash
docker exec -it primeGraph_postgres psql -U primeGraph -d primeGraph
```

### Accessing pgAdmin

pgAdmin is available at http://localhost:5050.

### Resetting the containers

```bash
# Stop all running containers
docker compose down

# Remove all containers, volumes, and images associated with the compose file

docker-compose down -v --rmi all

# For a thorough cleanup, you can also remove any dangling volumes

docker volume prune -f

# Starting the containers again

docker compose up
```

### Some postgresql guidance/interesting links

- [PostgreSQL Best Practices](https://wiki.postgresql.org/wiki/Don%27t_Do_This?&aid=recqNQC2McEJ8qXlo&_bhlid=b8182506d1c9ebc506f9897c68711ddc31426e2d#Don.27t_use_varchar.28n.29_by_default)

## Services

### PostgreSQL

The PostgreSQL database is available at:

- Host: `localhost`
- Port: `5432` (configurable in .env)
- Database: `primegraph`
- Username: `primegraph`
- Password: `primegraph`

### PgAdmin

The PgAdmin web interface is available at:

- URL: http://localhost:5050
- Email: `admin@primegraph.com`
- Password: `admin`

### Redis

The Redis service is available at:

- Host: `localhost`
- Port: `6379` (configurable in .env)
- No authentication by default

## Using Redis for Streaming

The Redis service can be used with primeGraph's streaming functionality:

```python
from primeGraph.graph.llm_clients import StreamingConfig, StreamingEventType

# Configure streaming with Redis
streaming_config = StreamingConfig(
    enabled=True,
    event_types={StreamingEventType.TEXT, StreamingEventType.CONTENT_BLOCK_STOP},
    redis_host="localhost",  # Docker-exposed Redis host
    redis_port=6379,         # Default Redis port
    redis_channel="my_stream_channel"  # Custom channel name
)
```

You can then consume these events from any service or application that can connect to Redis:

```python
import redis
import json

# Connect to Redis
r = redis.Redis(host="localhost", port=6379)
pubsub = r.pubsub()

# Subscribe to your stream channel
pubsub.subscribe("my_stream_channel")

# Process incoming messages
for message in pubsub.listen():
    if message["type"] == "message":
        event = json.loads(message["data"])
        print(f"Received event: {event['type']}")

        if event["type"] == "text":
            print(event["text"], end="", flush=True)
```
