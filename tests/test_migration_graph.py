"""Regression tests for the Alembic revision graph."""

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


ROOT = Path(__file__).resolve().parents[1]


def _migration_scripts() -> ScriptDirectory:
    """Load the repository's Alembic migration graph."""
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "db" / "migrations"))
    return ScriptDirectory.from_config(config)


def test_migration_graph_has_one_head() -> None:
    """Require ``alembic upgrade head`` to resolve to one revision."""
    assert _migration_scripts().get_heads() == ["011_merge_migration_heads"]


def test_released_per_app_revision_is_preserved() -> None:
    """Keep the revision identifier already stored by deployed databases."""
    revision = _migration_scripts().get_revision("008_per_app_config_and_audit")

    assert revision is not None
    assert revision.down_revision == "007_create_chat_feedback_table"
