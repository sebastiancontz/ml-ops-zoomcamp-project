build:
	docker-compose up --build --detach

kill:
	docker-compose down --remove-orphans --volumes
