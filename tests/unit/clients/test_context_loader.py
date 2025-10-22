"""
Tests for Context Loader

Tests context loading from local directories and GCS buckets.
"""

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from data_planning_agent.clients.context_loader import ContextLoader, load_context_from_directory

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture


def test_load_context_none_path() -> None:
    """Test that None path returns None."""
    loader = ContextLoader()
    result = loader.load_context(None)
    assert result is None


def test_load_context_empty_path() -> None:
    """Test that empty string path returns None."""
    loader = ContextLoader()
    result = loader.load_context("")
    assert result is None


def test_load_context_nonexistent_directory() -> None:
    """Test that nonexistent directory returns None."""
    loader = ContextLoader()
    result = loader.load_context("/nonexistent/directory")
    assert result is None


def test_load_context_empty_directory() -> None:
    """Test that empty directory returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = ContextLoader()
        result = loader.load_context(tmpdir)
        assert result is None


def test_load_context_single_file() -> None:
    """Test loading context from a single markdown file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a single context file
        context_file = Path(tmpdir) / "context.md"
        context_file.write_text("# Test Context\n\nThis is test context.")

        loader = ContextLoader()
        result = loader.load_context(tmpdir)

        assert result is not None
        assert "Test Context" in result
        assert "This is test context" in result
        assert "<!-- Context from: context.md -->" in result


def test_load_context_multiple_files_alphabetical() -> None:
    """Test that multiple files are loaded in alphabetical order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files in non-alphabetical order
        (Path(tmpdir) / "02_second.md").write_text("Second content")
        (Path(tmpdir) / "01_first.md").write_text("First content")
        (Path(tmpdir) / "03_third.md").write_text("Third content")

        loader = ContextLoader()
        result = loader.load_context(tmpdir)

        assert result is not None

        # Check that files appear in alphabetical order
        first_pos = result.find("First content")
        second_pos = result.find("Second content")
        third_pos = result.find("Third content")

        assert first_pos < second_pos < third_pos


def test_load_context_filters_non_markdown() -> None:
    """Test that non-markdown files are filtered out."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create markdown and non-markdown files
        (Path(tmpdir) / "context.md").write_text("Markdown content")
        (Path(tmpdir) / "readme.txt").write_text("Text file")
        (Path(tmpdir) / "data.json").write_text('{"key": "value"}')

        loader = ContextLoader()
        result = loader.load_context(tmpdir)

        assert result is not None
        assert "Markdown content" in result
        assert "Text file" not in result
        assert "key" not in result


def test_load_context_markdown_extension() -> None:
    """Test that .markdown extension is also supported."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "context.markdown").write_text("Markdown content")

        loader = ContextLoader()
        result = loader.load_context(tmpdir)

        assert result is not None
        assert "Markdown content" in result


def test_load_context_empty_files_ignored() -> None:
    """Test that empty or whitespace-only files are ignored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "empty.md").write_text("")
        (Path(tmpdir) / "whitespace.md").write_text("   \n\n  ")
        (Path(tmpdir) / "content.md").write_text("Actual content")

        loader = ContextLoader()
        result = loader.load_context(tmpdir)

        assert result is not None
        assert "Actual content" in result
        assert "<!-- Context from: empty.md -->" not in result
        assert "<!-- Context from: whitespace.md -->" not in result


def test_load_context_file_read_error(mocker: "MockerFixture") -> None:
    """Test graceful handling of file read errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "good.md").write_text("Good content")
        (Path(tmpdir) / "bad.md").write_text("Bad content")

        loader = ContextLoader()

        # Mock open to raise error for bad.md
        original_open = open

        def mock_open(path, *args, **kwargs):
            if "bad.md" in str(path):
                raise PermissionError("Permission denied")
            return original_open(path, *args, **kwargs)

        mocker.patch("builtins.open", side_effect=mock_open)

        result = loader.load_context(tmpdir)

        # Should still load good.md
        assert result is not None
        assert "Good content" in result


def test_load_context_gcs_path(mocker: "MockerFixture") -> None:
    """Test loading context from GCS bucket."""
    mock_storage_client = mocker.patch("google.cloud.storage.Client")
    mock_bucket = mocker.MagicMock()
    mock_blob1 = mocker.MagicMock()
    mock_blob2 = mocker.MagicMock()

    # Setup mock blobs
    mock_blob1.name = "context/01_first.md"
    mock_blob1.exists.return_value = True
    mock_blob1.download_as_text.return_value = "First content"

    mock_blob2.name = "context/02_second.md"
    mock_blob2.exists.return_value = True
    mock_blob2.download_as_text.return_value = "Second content"

    # Setup mock bucket
    mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
    mock_bucket.blob.side_effect = lambda path: (
        mock_blob1 if "01_first" in path else mock_blob2
    )

    mock_storage_client.return_value.bucket.return_value = mock_bucket

    loader = ContextLoader()
    result = loader.load_context("gs://my-bucket/context/")

    assert result is not None
    assert "First content" in result
    assert "Second content" in result


def test_load_context_from_directory_convenience() -> None:
    """Test the convenience function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "context.md").write_text("Test content")

        result = load_context_from_directory(tmpdir)

        assert result is not None
        assert "Test content" in result


def test_context_loader_separates_files() -> None:
    """Test that files are separated with blank lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "01.md").write_text("First")
        (Path(tmpdir) / "02.md").write_text("Second")

        loader = ContextLoader()
        result = loader.load_context(tmpdir)

        assert result is not None
        # Files should be separated by two newlines
        assert "\n\n" in result

