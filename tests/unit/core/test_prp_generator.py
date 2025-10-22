"""
Tests for PRP Generator

Tests Data PRP generation functionality.
"""

import pytest

from data_planning_agent.core.prp_generator import PRPGenerator
from data_planning_agent.models.session import PlanningSession


@pytest.mark.asyncio
async def test_generate_prp(
    prp_generator: PRPGenerator, planning_session: PlanningSession
) -> None:
    """Test generating a Data PRP from a session."""
    planning_session.is_complete = True

    prp_content, file_path = await prp_generator.generate_prp(
        session=planning_session, save_to_file=True
    )

    assert prp_content is not None
    assert "# Data Product Requirement Prompt" in prp_content
    assert file_path == "/test/path/data_prp.md"
    assert planning_session.data_prp_generated is True
    assert planning_session.data_prp_content == prp_content


@pytest.mark.asyncio
async def test_generate_prp_without_saving(
    prp_generator: PRPGenerator, planning_session: PlanningSession
) -> None:
    """Test generating a Data PRP without saving to file."""
    planning_session.is_complete = True

    prp_content, file_path = await prp_generator.generate_prp(
        session=planning_session, save_to_file=False
    )

    assert prp_content is not None
    assert file_path is None
    assert planning_session.data_prp_generated is True


@pytest.mark.asyncio
async def test_generate_prp_incomplete_session(
    prp_generator: PRPGenerator, planning_session: PlanningSession
) -> None:
    """Test generating a Data PRP from an incomplete session (should still work)."""
    planning_session.is_complete = False

    prp_content, file_path = await prp_generator.generate_prp(
        session=planning_session, save_to_file=False
    )

    assert prp_content is not None
    assert planning_session.data_prp_generated is True


def test_format_prp_summary(prp_generator: PRPGenerator) -> None:
    """Test formatting a PRP summary for display."""
    prp_content = "# Data Product Requirement Prompt\n\nTest content here"
    file_path = "/test/path/data_prp.md"

    summary = prp_generator.format_prp_summary(prp_content, file_path)

    assert "Data Product Requirement Prompt Generated" in summary
    assert file_path in summary
    assert "Preview" in summary

