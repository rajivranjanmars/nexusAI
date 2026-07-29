#!/usr/bin/env bash
# deploy.sh
# Bash script for deploying on Linux Production VM with Git metadata

# Exit immediately if a command exits with a non-zero status
set -e

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "Deploying LPUAI with Build Metadata:"
echo "  Branch:    ${BRANCH}"
echo "  Commit:    ${COMMIT}"
echo "  Timestamp: ${TIMESTAMP}"

export GIT_BRANCH="${BRANCH}"
export GIT_COMMIT_SHA="${COMMIT}"
export BUILD_TIMESTAMP="${TIMESTAMP}"

docker compose up -d --build
