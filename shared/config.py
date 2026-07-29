"""
Shared configuration module using Pydantic BaseSettings.

Single source of truth for all environment-driven configuration across the
Python codebase. Settings are grouped into logical sections: LLM, Database,
Redis, pgvector, Security, and Proxy.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables and ``.env``."""

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_base_url: str = Field(
        ...,
        description="Base URL of the OpenAI-compatible LLM provider",
    )
    llm_api_key: str = Field(
        ...,
        description="API key for the LLM provider",
    )
    llm_fast_model: str = Field(
        ...,
        description="Model identifier for fast and inexpensive tasks",
    )
    llm_smart_model: str = Field(
        ...,
        description="Model identifier for deeper reasoning tasks",
    )
    llm_max_tokens_fast: int = Field(
        default=500,
        description="Maximum response tokens for the fast model",
    )
    llm_max_tokens_smart: int = Field(
        default=600,
        description="Maximum response tokens for the smart model",
    )

# ── BuildInfo ────────────────────────────────────────────────────────────
    environment: str = Field(
        default="local",
        description="Deployment environment name (local | staging | production)",
    )
    git_branch: str = Field(
        default="unknown",
        description="Git branch name, injected at build/deploy time",
    )

    build_timestamp: str = Field(
    default="",
    description="ISO timestamp of the Docker image build",
    )

    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the embedding provider",
    )
    embedding_api_key: str | None = Field(
        default=None,
        description="API key for the embedding provider",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    embedding_provider: str = Field(
        default="remote",
        description="Embedding provider mode: remote, local, or auto",
    )
    embedding_local_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Local sentence-transformers model identifier",
    )

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = Field(
        ..., description="PostgreSQL connection URL for the primary application database"
    )

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL",
    )

    # ── pgvector / ChromaDB ──────────────────────────────────────────────────
    pgvector_url: str = Field(
        ...,
        description="PostgreSQL + pgvector connection URL",
    )
    chroma_db_dir: str = Field(
        default="./data/chroma_db",
        description="Directory for local ChromaDB storage",
    )

    # ── Security ─────────────────────────────────────────────────────────────
    jwt_secret: str = Field(..., description="Secret key for JWT signing")
    proxy_jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm used by the backend proxy",
    )
    sms_secret_key: str | None = Field(
        default=None,
        description=(
            "Fernet key(s) used to encrypt per-app SMS gateway credentials at rest. "
            "urlsafe-base64 32-byte key from cryptography.fernet.Fernet.generate_key(); "
            "supply a comma-separated list to rotate (first encrypts, all decrypt). "
            "MUST be distinct from jwt_secret and supplied via env/secret store, never the DB."
        ),
    )
    sms_allowed_gateway_hosts: str = Field(
        default="enterprise.smsgupshup.com",
        description=(
            "Comma-separated allowlist of hosts a per-app SMS api_url may point at. "
            "Prevents an admin-supplied api_url from turning the OTP send into an "
            "SSRF / credential+OTP exfiltration sink."
        ),
    )

    # ── MCP Server ───────────────────────────────────────────────────────────
    mcp_server_host: str = Field(
        default="0.0.0.0",
        description="Host interface for the MCP server",
    )
    mcp_server_port: int = Field(
        default=8000,
        description="Port for the MCP server (SSE transport)",
    )

    # ── Backend Proxy ────────────────────────────────────────────────────────
    proxy_host: str = Field(
        default="0.0.0.0",
        description="Host interface for the backend proxy",
    )
    proxy_port: int = Field(
        default=5123,
        description="Public port exposed by the backend proxy",
    )
    proxy_allowed_origins: str = Field(
        default="*",
        description="Comma-separated CORS origins for browser clients",
    )
    proxy_token_issuer: str = Field(
        default="LPUAI-backend-proxy",
        description="Issuer claim used for backend proxy JWTs",
    )
    proxy_access_token_ttl_minutes: int = Field(
        default=15,
        description="Access token lifetime issued by the backend proxy",
    )
    proxy_refresh_token_ttl_hours: int = Field(
        default=24,
        description="Refresh token lifetime issued by the backend proxy",
    )
    proxy_rate_limit_requests: int = Field(
        default=30,
        description="Maximum requests allowed per rate-limit window",
    )
    proxy_rate_limit_window_seconds: int = Field(
        default=60,
        description="Rate-limit window size in seconds",
    )
    proxy_login_password: str = Field(
        default="change-me-proxy-login",
        description="Bootstrap student login password for local proxy auth",
    )
    proxy_admin_login_password: str = Field(
        default="change-me-admin-login",
        description="Bootstrap admin login password for privileged proxy auth",
    )
    proxy_mcp_api_key: str = Field(
        default="LPUAI_backend_proxy_global_McpProxyInternalKey1234",
        description="Internal API key used by the backend proxy to call MCP",
    )
    proxy_mcp_sse_url: str = Field(
        default="http://mcp_server:8000/sse",
        description="Private SSE endpoint for downstream MCP access",
    )
    proxy_mcp_http_timeout_seconds: int = Field(
        default=10,
        description="HTTP timeout for downstream MCP requests",
    )
    proxy_mcp_sse_read_timeout_seconds: int = Field(
        default=300,
        description="Read timeout for downstream MCP SSE sessions",
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(
        default="info",
        description="Logging level (debug | info | warning | error)",
    )

    # ── Alerting ─────────────────────────────────────────────────────────────
    slack_webhook_url: str | None = Field(
        default=None,
        description="Slack Incoming Webhook URL for ERROR/CRITICAL alerts. Leave unset to disable.",
    )
    alert_rate_limit_seconds: int = Field(
        default=60,
        description="Min seconds between repeated Slack alerts for the same error fingerprint.",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


# Singleton - import ``settings`` wherever needed.
settings = Settings()  # type: ignore[call-arg]
