"""
Context Loader

Loads organizational context from markdown files stored locally or in GCS.
Context is prepended to all AI prompts to customize agent behavior.
"""

import logging
import os
from typing import Optional

from google.cloud import storage

logger = logging.getLogger(__name__)


class ContextLoader:
    """
    Loads context from a directory of markdown files.

    Supports both local filesystem and GCS bucket paths.
    """

    def __init__(self):
        """Initialize the context loader."""
        self.gcs_client: Optional[storage.Client] = None

    def _ensure_gcs_client(self) -> storage.Client:
        """
        Ensure GCS client is initialized.

        Returns:
            Initialized GCS client
        """
        if self.gcs_client is None:
            self.gcs_client = storage.Client()
            logger.debug("Initialized GCS client for context loading")
        return self.gcs_client

    def _list_local_markdown_files(self, dir_path: str) -> list[str]:
        """
        List markdown files in a local directory.

        Args:
            dir_path: Local directory path

        Returns:
            List of markdown filenames (sorted alphabetically)
        """
        try:
            if not os.path.exists(dir_path):
                logger.warning(f"Context directory does not exist: {dir_path}")
                return []

            if not os.path.isdir(dir_path):
                logger.warning(f"Context path is not a directory: {dir_path}")
                return []

            files = [
                f
                for f in os.listdir(dir_path)
                if f.endswith((".md", ".markdown")) and os.path.isfile(os.path.join(dir_path, f))
            ]

            return sorted(files)

        except Exception as e:
            logger.error(f"Error listing local context files: {e}", exc_info=True)
            return []

    def _list_gcs_markdown_files(self, bucket_name: str, prefix: str) -> list[str]:
        """
        List markdown files in a GCS bucket directory.

        Args:
            bucket_name: GCS bucket name
            prefix: Directory prefix within bucket

        Returns:
            List of markdown filenames (sorted alphabetically)
        """
        try:
            gcs_client = self._ensure_gcs_client()
            bucket = gcs_client.bucket(bucket_name)

            # List blobs with prefix
            blobs = bucket.list_blobs(prefix=prefix)

            # Filter for markdown files (not subdirectories)
            files = []
            for blob in blobs:
                # Get filename relative to prefix
                relative_path = blob.name[len(prefix) :].lstrip("/")

                # Skip if it's a subdirectory or empty
                if not relative_path or "/" in relative_path:
                    continue

                # Check if markdown file
                if relative_path.endswith((".md", ".markdown")):
                    files.append(relative_path)

            return sorted(files)

        except Exception as e:
            logger.error(f"Error listing GCS context files: {e}", exc_info=True)
            return []

    def _read_local_file(self, file_path: str) -> Optional[str]:
        """
        Read content from a local file.

        Args:
            file_path: Path to local file

        Returns:
            File content or None if read fails
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read local context file {file_path}: {e}")
            return None

    def _read_gcs_file(self, bucket_name: str, blob_path: str) -> Optional[str]:
        """
        Read content from a GCS blob.

        Args:
            bucket_name: GCS bucket name
            blob_path: Blob path within bucket

        Returns:
            File content or None if read fails
        """
        try:
            gcs_client = self._ensure_gcs_client()
            bucket = gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            if not blob.exists():
                logger.warning(f"GCS context file does not exist: gs://{bucket_name}/{blob_path}")
                return None

            return blob.download_as_text(encoding="utf-8").strip()

        except Exception as e:
            logger.warning(f"Failed to read GCS context file gs://{bucket_name}/{blob_path}: {e}")
            return None

    def load_context(self, dir_path: Optional[str]) -> Optional[str]:
        """
        Load organizational context from a directory of markdown files.

        Supports both local directories and GCS bucket paths.

        Args:
            dir_path: Directory path (local or gs://bucket/path)
                     If None or empty, returns None

        Returns:
            Concatenated context from all markdown files, or None if no context
        """
        if not dir_path:
            logger.debug("No context directory configured")
            return None

        # Detect path type
        is_gcs = dir_path.startswith("gs://")

        logger.info(f"Loading context from: {dir_path}")

        # List markdown files
        if is_gcs:
            # Parse GCS path
            path_parts = dir_path[5:].split("/", 1)
            bucket_name = path_parts[0]
            prefix = path_parts[1] if len(path_parts) > 1 else ""

            if not prefix.endswith("/"):
                prefix += "/"

            md_files = self._list_gcs_markdown_files(bucket_name, prefix)
        else:
            md_files = self._list_local_markdown_files(dir_path)

        if not md_files:
            logger.warning(f"No markdown files found in context directory: {dir_path}")
            return None

        logger.info(f"Found {len(md_files)} context file(s): {', '.join(md_files)}")

        # Read and concatenate files
        content_parts = []
        for filename in md_files:
            # Read file content
            if is_gcs:
                blob_path = f"{prefix}{filename}"
                content = self._read_gcs_file(bucket_name, blob_path)
            else:
                file_path = os.path.join(dir_path, filename)
                content = self._read_local_file(file_path)

            # Add to parts if content exists
            if content:
                # Add comment marker to identify source file
                content_parts.append(f"<!-- Context from: {filename} -->\n{content}")
                logger.debug(f"Loaded context from {filename} ({len(content)} chars)")

        if not content_parts:
            logger.warning("All context files were empty or unreadable")
            return None

        # Concatenate with blank lines between files
        combined_context = "\n\n".join(content_parts)

        logger.info(
            f"Successfully loaded context from {len(content_parts)} file(s) "
            f"({len(combined_context)} total chars)"
        )

        return combined_context


def load_context_from_directory(dir_path: Optional[str]) -> Optional[str]:
    """
    Load organizational context from a directory (convenience function).

    Args:
        dir_path: Directory path (local or GCS)

    Returns:
        Combined context string or None
    """
    loader = ContextLoader()
    return loader.load_context(dir_path)

