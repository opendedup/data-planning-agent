"""
Storage Client

Client for writing Data PRPs to different storage backends (GCS or local filesystem).
"""

import logging
import os
from pathlib import Path
from typing import Optional

from google.cloud import storage

logger = logging.getLogger(__name__)


class StorageClient:
    """
    Client for writing files to GCS or local storage.

    Supports:
    - GCS paths: gs://bucket-name/path/to/file.md
    - Local paths: /absolute/path/file.md or ./relative/path/file.md
    - File URI: file:///absolute/path/file.md
    """

    def __init__(self, default_output_dir: str = "./output"):
        """
        Initialize storage client.

        Args:
            default_output_dir: Default directory for output files
        """
        self.default_output_dir = default_output_dir
        self.gcs_client: Optional[storage.Client] = None

        logger.info(f"Initialized storage client (default: {default_output_dir})")

    def _ensure_gcs_client(self) -> storage.Client:
        """
        Ensure GCS client is initialized.

        Returns:
            Initialized GCS client
        """
        if self.gcs_client is None:
            self.gcs_client = storage.Client()
            logger.debug("Initialized GCS client")
        return self.gcs_client

    def _parse_path(self, path: str) -> tuple[str, str, str]:
        """
        Parse a path and determine its type.

        Args:
            path: Path to parse

        Returns:
            Tuple of (path_type, bucket_or_dir, file_path)
            - path_type: 'gcs' or 'local'
            - bucket_or_dir: Bucket name (GCS) or directory path (local)
            - file_path: File path within bucket or directory
        """
        # Handle file:// URI
        if path.startswith("file://"):
            path = path[7:]  # Remove file:// prefix

        # Check if GCS path
        if path.startswith("gs://"):
            path = path[5:]  # Remove gs:// prefix
            parts = path.split("/", 1)
            bucket_name = parts[0]
            file_path = parts[1] if len(parts) > 1 else ""
            return ("gcs", bucket_name, file_path)

        # Local path
        path_obj = Path(path).expanduser().resolve()
        directory = str(path_obj.parent)
        filename = path_obj.name
        return ("local", directory, filename)

    def write_file(self, content: str, output_path: Optional[str] = None) -> str:
        """
        Write content to a file (GCS or local).

        Args:
            content: The content to write
            output_path: Path where to write the file (optional)
                        If None, uses default_output_dir with generated filename

        Returns:
            The full path where the file was written

        Raises:
            Exception: If write fails
        """
        # Generate default path if not provided
        if output_path is None:
            from datetime import datetime

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"data_prp_{timestamp}.md"

            if self.default_output_dir.startswith("gs://"):
                output_path = f"{self.default_output_dir.rstrip('/')}/{filename}"
            else:
                output_path = str(Path(self.default_output_dir) / filename)

        # Parse the path
        path_type, bucket_or_dir, file_path = self._parse_path(output_path)

        try:
            if path_type == "gcs":
                return self._write_to_gcs(content, bucket_or_dir, file_path)
            else:
                return self._write_to_local(content, bucket_or_dir, file_path)
        except Exception as e:
            logger.error(f"Error writing file to {output_path}: {e}", exc_info=True)
            raise

    def _write_to_gcs(self, content: str, bucket_name: str, blob_path: str) -> str:
        """
        Write content to Google Cloud Storage.

        Args:
            content: Content to write
            bucket_name: GCS bucket name
            blob_path: Path within bucket

        Returns:
            Full GCS URI (gs://bucket/path)
        """
        gcs_client = self._ensure_gcs_client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        blob.upload_from_string(content, content_type="text/markdown")
        full_path = f"gs://{bucket_name}/{blob_path}"

        logger.info(f"Wrote file to GCS: {full_path}")
        return full_path

    def _write_to_local(self, content: str, directory: str, filename: str) -> str:
        """
        Write content to local filesystem.

        Args:
            content: Content to write
            directory: Directory path
            filename: Filename

        Returns:
            Full local path
        """
        # Create directory if it doesn't exist
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Write file
        file_path = dir_path / filename
        file_path.write_text(content, encoding="utf-8")

        full_path = str(file_path)
        logger.info(f"Wrote file to local: {full_path}")
        return full_path

    def read_file(self, file_path: str) -> str:
        """
        Read content from a file (GCS or local).

        Args:
            file_path: Path to read from

        Returns:
            File content

        Raises:
            Exception: If read fails
        """
        path_type, bucket_or_dir, path = self._parse_path(file_path)

        try:
            if path_type == "gcs":
                gcs_client = self._ensure_gcs_client()
                bucket = gcs_client.bucket(bucket_or_dir)
                blob = bucket.blob(path)
                content = blob.download_as_text(encoding="utf-8")
                logger.debug(f"Read file from GCS: gs://{bucket_or_dir}/{path}")
                return content
            else:
                file_path_obj = Path(bucket_or_dir) / path
                content = file_path_obj.read_text(encoding="utf-8")
                logger.debug(f"Read file from local: {file_path_obj}")
                return content
        except Exception as e:
            logger.error(f"Error reading file from {file_path}: {e}", exc_info=True)
            raise

