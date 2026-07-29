
.PHONY: test test-unit test-integration test-docker test-simple test-ci clean-test

# Default test target
test: test-unit test-integration

# Run unit tests only
test-unit:
	@echo "Running unit tests..."
	python -m pytest scripts/tests/test_unit.py -v

# Run integration tests (requires Docker)
test-integration:
	@echo "Running integration tests..."
	docker-compose up -d
	sleep 30
	python scripts/tests/run_tests.py local --skip-docker
	docker-compose down

# Run full Docker-based tests
test-docker:
	@echo "Running Docker-based tests..."
	./scripts/tests/test_docker_compose.sh

# Run simple tests (no Docker required)
test-simple:
	@echo "Running simple tests..."
	python scripts/tests/simple_test.py

# Run CI tests
test-ci:
	@echo "Running CI tests..."
	./scripts/tests/run_ci_tests.sh

# Clean test artifacts
clean-test:
	@echo "Cleaning test artifacts..."
	rm -f test_report.json ci_test_results.json
	docker-compose down --volumes --remove-orphans
	docker system prune -f

# Build test images
build-test:
	@echo "Building test images..."
	docker-compose build

# Start test services
start-test:
	@echo "Starting test services..."
	docker-compose up -d

# Stop test services
stop-test:
	@echo "Stopping test services..."
	docker-compose down

# Show test logs
logs-test:
	@echo "Showing test logs..."
	docker-compose logs --tail=100
deploy:
	GIT_BRANCH=$$(git rev-parse --abbrev-ref HEAD) \
	GIT_COMMIT_SHA=$$(git rev-parse --short HEAD) \
	BUILD_TIMESTAMP=$$(date -u +%Y-%m-%dT%H:%M:%SZ) \
	docker compose up -d --build