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
