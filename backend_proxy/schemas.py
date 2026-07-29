"""
Pydantic request and response models for the backend proxy API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def normalize_utc_datetime(value: Optional[datetime]) -> Optional[datetime]:
    """Normalize one datetime to an aware UTC value.

    Naive datetimes are treated as already being in UTC.
    """

    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class AppTokenRequest(BaseModel):
    """Request body for issuing tokens via App Registry signed JWT."""

    signed_token: str = Field(..., min_length=1, description="Application signed JWT")


class RefreshTokenRequest(BaseModel):
    """Request body for refreshing an access token."""

    refresh_token: str = Field(..., min_length=1, description="Signed refresh JWT")


class AppTokenResponse(BaseModel):
    """Response payload for app-level token issuance."""

    access_token: str
    refresh_token: str
    expires_in: int
    refresh_expires_in: int
    token_type: str = "Bearer"
    app_id: str
    app_name: str
    role: str
    app_config: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    force_workflow: Optional[str] = None


class RefreshTokenResponse(BaseModel):
    """Response payload for refreshing an access token."""

    access_token: str
    expires_in: int
    token_type: str = "Bearer"


def _normalize_indian_mobile(value: str) -> str:
    """Normalize input to a bare 10-digit Indian mobile number.

    Raises:
        ValueError: If the value is not a valid 10-digit number.
    """
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) != 10:
        raise ValueError("Enter a valid 10-digit mobile number.")
    return digits


class OtpRequestRequest(BaseModel):
    """Request body to send an OTP to a mobile number."""

    phone: str = Field(..., description="Subscriber mobile number (10-digit Indian).")

    @field_validator("phone")
    @classmethod
    def _validate_phone(cls, value: str) -> str:
        return _normalize_indian_mobile(value)


class OtpRequestResponse(BaseModel):
    """Response payload after an OTP has been dispatched."""

    success: bool = True
    expires_in: int = Field(description="OTP validity window in seconds.")
    resend_after: int = Field(description="Seconds before a new OTP may be requested.")


class OtpVerifyRequest(BaseModel):
    """Request body to verify an OTP and upgrade a guest token."""

    phone: str = Field(..., description="Mobile number the OTP was sent to.")
    otp: str = Field(..., min_length=1, description="One-time password entered by the user.")

    @field_validator("phone")
    @classmethod
    def _validate_phone(cls, value: str) -> str:
        return _normalize_indian_mobile(value)

    @field_validator("otp")
    @classmethod
    def _normalize_otp(cls, value: str) -> str:
        digits = "".join(ch for ch in str(value or "") if ch.isdigit())
        if not digits:
            raise ValueError("Enter the numeric OTP.")
        return digits


class OtpVerifyResponse(BaseModel):
    """Response payload returning phone-verified access/refresh tokens."""

    access_token: str
    refresh_token: str
    expires_in: int
    refresh_expires_in: int
    token_type: str = "Bearer"
    phone: str
    phone_verified: bool = True


class ChatMessageRequest(BaseModel):
    """Request body for the secure chat endpoint."""

    message: str = Field(..., min_length=1, description="Student chat message")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional stable conversation/session identifier for guest or multi-turn chats.",
    )
    stream: bool = Field(
        default=False,
        description="Reserved for clients that want streaming semantics. Use /api/chat/message/stream.",
    )


class ChatMessageResponse(BaseModel):
    """Response payload for the secure chat endpoint."""

    response: str
    workflow: str
    session_id: Optional[str] = None
    cache_hit: bool = False
    lead_progress: Optional[Dict[str, Any]] = None
    helper_buttons: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Frontend helper buttons derived from the workflow progress.",
    )
    token_usage: Dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregated token usage reported by the orchestration pipeline.",
    )


class ChatFeedbackRequest(BaseModel):
    """Request body for storing feedback on a chat response."""

    session_id: str = Field(..., min_length=1, description="Conversation/session identifier.")
    feedback_flag: bool = Field(..., description="Whether the feedback is positive.")
    feedback_text: Optional[str] = Field(
        default=None,
        description="Optional free-form feedback text.",
    )
    user_message: str = Field(..., min_length=1, description="The user message tied to the feedback.")
    response: str = Field(..., min_length=1, description="The assistant response tied to the feedback.")

    @field_validator("session_id", "user_message", "response")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field must not be blank.")
        return normalized

    @field_validator("feedback_text")
    @classmethod
    def _normalize_optional_feedback_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ChatFeedbackResponse(BaseModel):
    """Response payload for chat feedback submission."""

    success: bool = True
    id: str


class FeedbackListQuery(BaseModel):
    """Query parameters for listing stored feedback records."""

    page: int = Field(default=1, ge=1, description="1-based page number for pagination mode.")
    page_size: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Page size for pagination mode.",
    )
    last: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="Return the most recent N records instead of paginated results.",
    )
    actor_id: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Optional exact actor identifier filter.",
    )
    start_date: Optional[datetime] = Field(
        default=None,
        description="Optional inclusive UTC lower bound on created_at.",
    )
    end_date: Optional[datetime] = Field(
        default=None,
        description="Optional inclusive UTC upper bound on created_at.",
    )

    @field_validator("actor_id")
    @classmethod
    def _normalize_actor_id(cls, value: Optional[str]) -> Optional[str]:
        """Strip optional actor filter values and collapse blank strings to None."""

        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("start_date", "end_date")
    @classmethod
    def _normalize_date_filters(cls, value: Optional[datetime]) -> Optional[datetime]:
        """Normalize optional date filters to UTC before validation or querying."""

        return normalize_utc_datetime(value)

    def validate_date_range(self) -> None:
        """Ensure normalized date filters form a valid inclusive range."""

        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be less than or equal to end_date")


class FeedbackRecordResponse(BaseModel):
    """Serialized feedback row returned by read APIs."""

    id: str
    session_id: str
    app_id: str
    actor_id: str
    feedback_flag: bool
    feedback_text: Optional[str] = None
    user_message: str
    response: str
    created_at: Optional[str] = None


class FeedbackListResponse(BaseModel):
    """Paginated feedback list response payload."""

    items: List[FeedbackRecordResponse]
    page: int
    page_size: int
    total: int


class ToolCallRequest(BaseModel):
    """Request body for allowlisted MCP tool execution."""

    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments forwarded to the downstream MCP tool",
    )


class ToolCallResponse(BaseModel):
    """Response payload for allowlisted MCP tool execution."""

    tool: str
    data: Any


class AppRegistrationRequest(BaseModel):
    """Request body for registering a new application."""

    name: str = Field(..., min_length=2, description="Human readable application name")
    domain: str = Field(
        ...,
        description="Allowed origin(s) for requests. Supports comma-separated values.",
    )
    public_key: str = Field(..., description="PEM formatted ES256 Public Key")
    allowed_workflows: List[str] = Field(
        default_factory=lambda: ["general"],
        description="Workflows this app is permitted to invoke"
    )
    allowed_tools: List[str] = Field(
        default_factory=lambda: ["detect_workflow", "get_student"],
        description="MCP Tools this app is permitted to use"
    )


class AppRegistrationResponse(BaseModel):
    """Response payload for app registration."""

    app_id: str
    name: str
    status: str = "Registered successfully"


class AppBootstrapRegistrationRequest(BaseModel):
    """Admin request body for one-shot PKI app bootstrap."""

    name: str = Field(..., min_length=2, description="Human readable application name")
    domain: str = Field(
        default="",
        description="Allowed origin(s) for requests. Supports comma-separated values. Leave empty for internal use.",
    )
    allowed_workflows: List[str] = Field(
        default_factory=lambda: ["general"],
        description="Workflows this app is permitted to invoke",
    )
    allowed_tools: List[str] = Field(
        default_factory=lambda: ["detect_workflow", "get_student", "chat_complete"],
        description="MCP tools this app is permitted to use",
    )


class AppBootstrapRegistrationResponse(BaseModel):
    """Response payload for one-shot PKI app bootstrap."""

    app_id: str
    name: str
    domain: str
    public_key: str
    private_key: str
    allowed_workflows: List[str]
    allowed_tools: List[str]
    proxy_url: str
    status: str = "Registered successfully"


class AppUserTokenMintRequest(BaseModel):
    """Request body to mint app-user tokens from a signed app JWT."""

    signed_token: str = Field(
        ...,
        min_length=1,
        description="ES256-signed app JWT minted by the application backend.",
    )
    force_workflow: Optional[str] = Field(
        default=None,
        description="Optional workflow to lock for this session (e.g. 'lead_capture'). Must be in allowed_workflows.",
    )


class DevAppTokenMintRequest(BaseModel):
    """Development-only request body to mint app-user tokens from a private key."""

    app_id: str = Field(..., min_length=1, description="Registered application ID")
    private_key: str = Field(..., min_length=1, description="PEM formatted ES256 private key")
    actor_id: str = Field(
        default="",
        description="Optional actor identifier (student_id, staff_id, etc.)",
    )
    actor_type: str = Field(
        default="student",
        description="Actor type: student, staff, customer, system, admin",
    )
    origin: str = Field(
        default="",
        description="Optional origin/domain to enforce domain matching during exchange",
    )
    jwt_ttl_seconds: int = Field(
        default=120,
        ge=30,
        le=3600,
        description="Expiry for the intermediate app-signed JWT",
    )
    force_workflow: Optional[str] = Field(
        default=None,
        description="Optional workflow to lock for this session (e.g. 'lead_capture'). Must be in allowed_workflows.",
    )


class DevAdminTokenMintRequest(BaseModel):
    """Development-only request body to mint an admin token from a private key."""

    app_id: str = Field(..., min_length=1, description="Registered admin application ID")
    private_key: str = Field(..., min_length=1, description="PEM formatted ES256 private key")
    actor_id: str = Field(
        default="ADMIN001",
        description="Optional admin actor identifier to embed in the issued proxy token",
    )
    origin: str = Field(
        default="",
        description="Optional origin/domain to enforce domain matching during exchange",
    )
    jwt_ttl_seconds: int = Field(
        default=120,
        ge=30,
        le=3600,
        description="Expiry for the intermediate app-signed JWT",
    )


class DevSignedTokenRequest(BaseModel):
    """Development-only request to generate a raw ES256 app-signed JWT."""

    app_id: str = Field(..., min_length=1, description="Registered application ID")
    private_key: str = Field(..., min_length=1, description="PEM formatted ES256 private key")
    actor_id: str = Field(
        default="",
        description="Optional actor identifier (student_id, staff_id, etc.)",
    )
    actor_type: str = Field(
        default="student",
        description="Actor type: student, staff, customer, system, admin",
    )
    jwt_ttl_seconds: int = Field(
        default=120,
        ge=30,
        le=3600,
        description="Expiry for the generated app-signed JWT",
    )


class DevSignedTokenResponse(BaseModel):
    """Response payload containing a raw ES256 app-signed JWT."""

    signed_token: str
    expires_in: int = Field(description="JWT lifetime in seconds")
    app_id: str


class AdminAppSummaryResponse(BaseModel):
    """Summary view of a registered application for admin tooling."""

    app_id: str
    name: str
    domain: str
    allowed_workflows: List[str]
    allowed_tools: List[str]
    rate_limit_rpm: int
    cache_ttl_seconds: int
    otp_enabled: bool
    caching_enabled: bool
    otp_length: int
    otp_ttl_seconds: int
    otp_max_attempts: int
    otp_resend_cooldown_seconds: int
    otp_max_requests_per_hour: int
    sms_daily_cap: int
    is_active: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AdminAppDetailResponse(AdminAppSummaryResponse):
    """Detailed admin view of a registered application."""

    public_key: str
    llm_model_override: Optional[str] = None
    token_quota_monthly: Optional[int] = None
    app_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    # Whether encrypted SMS credentials are configured (never the values).
    sms_credentials_configured: bool = False


class AdminAppUpdateRequest(BaseModel):
    """Patchable fields for an application.

    OTP behavior knobs carry the same hard bounds the runtime clamps to, so an
    out-of-range value is rejected at the API boundary (422) rather than being
    silently clamped. SMS gateway credentials are NOT patchable here — they are
    secrets and go through the dedicated write-only ``/sms-credentials`` endpoint.
    """

    name: Optional[str] = Field(default=None, min_length=2)
    domain: Optional[str] = Field(
        default=None,
        description="Allowed origin(s), comma-separated if multiple.",
    )
    allowed_workflows: Optional[List[str]] = None
    allowed_tools: Optional[List[str]] = None
    rate_limit_rpm: Optional[int] = Field(default=None, ge=1)
    cache_ttl_seconds: Optional[int] = Field(default=None, ge=0)
    token_quota_monthly: Optional[int] = Field(default=None, ge=0)
    llm_model_override: Optional[str] = None
    app_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    otp_enabled: Optional[bool] = Field(
        default=None, description="Enable guest phone/OTP verification for this app.",
    )
    caching_enabled: Optional[bool] = Field(
        default=None, description="Enable semantic response caching for this app.",
    )
    otp_length: Optional[int] = Field(default=None, ge=6, le=8)
    otp_ttl_seconds: Optional[int] = Field(default=None, ge=60, le=600)
    otp_max_attempts: Optional[int] = Field(default=None, ge=3, le=10)
    otp_resend_cooldown_seconds: Optional[int] = Field(default=None, ge=30, le=600)
    otp_max_requests_per_hour: Optional[int] = Field(default=None, ge=1, le=20)
    sms_daily_cap: Optional[int] = Field(default=None, ge=1, le=100_000)
    is_active: Optional[bool] = None


class SmsCredentialsRequest(BaseModel):
    """Write-only per-app SMS gateway credentials (encrypted before storage)."""

    user_id: str = Field(..., min_length=1, description="SMS gateway account user id.")
    password: str = Field(..., min_length=1, description="SMS gateway account password.")
    api_url: Optional[str] = Field(
        default=None, description="Gateway URL; platform default used when omitted.",
    )
    message_template: Optional[str] = Field(
        default=None,
        description="OTP SMS copy; must contain exactly one {otp} and no other braces.",
    )

    @field_validator("api_url")
    @classmethod
    def _validate_api_url(cls, value: Optional[str]) -> Optional[str]:
        # Reject SSRF / exfiltration sinks (non-https, non-allowlisted host,
        # private/loopback/link-local) before the credentials are ever stored.
        from orchestration.sms_service import GatewayUrlError, validate_sms_api_url

        try:
            return validate_sms_api_url(value)
        except GatewayUrlError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("message_template")
    @classmethod
    def _validate_message_template(cls, value: Optional[str]) -> Optional[str]:
        # Enforce the safe-template contract (exactly one {otp}, no other braces)
        # at the same layer as api_url so both fields reject with a 422.
        if value is None:
            return None
        from orchestration.sms_service import TemplateError, validate_message_template

        try:
            return validate_message_template(value)
        except TemplateError as exc:
            raise ValueError(str(exc)) from exc


class SmsCredentialsResponse(BaseModel):
    """Masked confirmation that per-app SMS credentials were stored."""

    configured: bool = True
    user_id_hint: str = Field(description="Masked user id (last 4 chars only).")


class AppAuditLogEntryResponse(BaseModel):
    """One recorded change to an app's configuration."""

    id: str
    app_id: str
    changed_by: str
    action: str
    changes: Dict[str, Any]
    created_at: Optional[str] = None


# ── Admin Dashboard Auth Models ──────────────────────────────────────────────


class AdminLoginRequest(BaseModel):
    """Admin dashboard login credentials."""
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class AdminUserProfile(BaseModel):
    """Public profile of an authenticated admin user."""
    id: str
    email: str
    display_name: str
    admin_role: str
    app_id: Optional[str] = None


class AdminLoginResponse(BaseModel):
    """Response payload for admin login."""
    access_token: str
    refresh_token: str
    expires_in: int
    refresh_expires_in: int
    token_type: str = "Bearer"
    admin_user: AdminUserProfile


class AdminRefreshRequest(BaseModel):
    """Request body for refreshing an admin access token."""
    refresh_token: str = Field(..., min_length=1)


class AdminRefreshResponse(BaseModel):
    """Response payload for admin token refresh."""
    access_token: str
    expires_in: int
    token_type: str = "Bearer"


# ── Admin User Management Models ─────────────────────────────────────────────


class AdminUserSummary(BaseModel):
    """Summary view of an admin user for list endpoints."""
    id: str
    email: str
    display_name: str
    role: str
    app_id: Optional[str] = None
    is_active: bool
    last_login_at: Optional[str] = None
    created_at: Optional[str] = None


class AdminUserCreateRequest(BaseModel):
    """Request body to create a new admin user."""
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=8)
    display_name: str = Field(..., min_length=1)
    role: str = Field(default="app_admin")
    app_id: Optional[str] = None


class AdminUserUpdateRequest(BaseModel):
    """Patchable fields for an admin user."""
    display_name: Optional[str] = Field(default=None, min_length=1)
    role: Optional[str] = None
    app_id: Optional[str] = None
    is_active: Optional[bool] = None
# ── Admin Observability Response Models ─────────────────────────────────────


class TokenUsageByModelItem(BaseModel):
    model: str
    calls: int
    tokens: int
    cost_usd: float


class TokenUsageByActionItem(BaseModel):
    action: str
    calls: int
    tokens: int


class TokenUsageSummaryResponse(BaseModel):
    """Aggregated token usage summary."""

    period: str
    total_calls: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    by_model: List[TokenUsageByModelItem]
    by_action: List[TokenUsageByActionItem]


class TokenUsageDailyItem(BaseModel):
    date: str
    calls: int
    tokens: int
    cost_usd: float


class TokenUsageByAppResponse(BaseModel):
    """Daily token usage breakdown for a specific app."""

    app_id: str
    period: str
    daily: List[TokenUsageDailyItem]


class TokenUsageDetailItem(BaseModel):
    id: str
    timestamp: Optional[str] = None
    app_id: Optional[str] = None
    actor_id: Optional[str] = None
    actor_type: Optional[str] = None
    action: Optional[str] = None
    model: Optional[str] = None
    model_tier: Optional[str] = None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: Optional[float] = None
    cache_hit: bool = False
    latency_ms: Optional[int] = None
    correlation_id: Optional[str] = None


class TokenUsageRecentResponse(BaseModel):
    """Paginated list of recent LLM invocations."""

    total: int
    limit: int
    offset: int
    items: List[TokenUsageDetailItem]


class RagAppStatsItem(BaseModel):
    app_id: str
    chunk_count: int
    source_types: Dict[str, int] = Field(default_factory=dict)
    source_count: int = 0


class RagStatsResponse(BaseModel):
    """Global or per-app RAG knowledge store statistics."""

    total_chunks: int
    total_apps: int
    apps: List[RagAppStatsItem] = Field(default_factory=list)


class CacheStatsResponse(BaseModel):
    """Semantic response cache statistics."""

    total_entries: int
    active_entries: int
    total_hits: int
    top_cached: List[Dict[str, Any]] = Field(default_factory=list)


class LlmHealthResponse(BaseModel):
    """LLM provider health check result."""

    status: str
    model: str
    latency_ms: int
    error: Optional[str] = None


class CostMapResponse(BaseModel):
    """Active LLM cost configuration."""

    cost_map: Dict[str, Dict[str, float]]
