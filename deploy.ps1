# deploy.ps1
# PowerShell script for deploying locally on Windows with Git metadata

$branch = git rev-parse --abbrev-ref HEAD
$commit = git rev-parse --short HEAD
$timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

Write-Host "Deploying LPUAI with Build Metadata:" -ForegroundColor Cyan
Write-Host "  Branch:    $branch"
Write-Host "  Commit:    $commit"
Write-Host "  Timestamp: $timestamp"

$env:GIT_BRANCH = $branch
$env:GIT_COMMIT_SHA = $commit
$env:BUILD_TIMESTAMP = $timestamp

docker compose up -d --build
