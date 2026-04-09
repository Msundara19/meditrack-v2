"""
Storage service — abstracts local vs Supabase Storage.

If SUPABASE_URL and SUPABASE_KEY are set, images are uploaded to
Supabase Storage and the public URL is returned.

If not set (local dev), images are saved to the local UPLOAD_DIR
and the URL is a relative API path (/api/wounds/{scan_id}/image).
"""
import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

_supabase_client = None


def _get_supabase():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        _supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    return _supabase_client


def is_cloud_storage() -> bool:
    return bool(settings.SUPABASE_URL and settings.SUPABASE_KEY)


def upload_image(content: bytes, filename: str, scan_id: str) -> str:
    """
    Upload image bytes and return the URL where it can be accessed.

    Returns:
        str: Public URL (Supabase) or local file path (dev)
    """
    if is_cloud_storage():
        return _upload_to_supabase(content, filename, scan_id)
    else:
        return _save_locally(content, filename)


def upload_annotated_image(image_array, scan_id: str, suffix: str) -> str:
    """
    Upload an annotated (OpenCV) image array.

    Returns:
        str: Public URL (Supabase) or local file path (dev)
    """
    import cv2
    _, buffer = cv2.imencode(".jpg", image_array)
    content = buffer.tobytes()
    filename = f"{scan_id}_annotated{suffix}"
    return upload_image(content, filename, scan_id)


def _upload_to_supabase(content: bytes, filename: str, scan_id: str) -> str:
    sb = _get_supabase()
    bucket = settings.SUPABASE_BUCKET
    path = f"{scan_id}/{filename}"

    sb.storage.from_(bucket).upload(
        path,
        content,
        {"content-type": "image/jpeg", "upsert": "true"},
    )

    public_url = sb.storage.from_(bucket).get_public_url(path)
    logger.info(f"Uploaded to Supabase: {public_url}")
    return public_url


def _save_locally(content: bytes, filename: str) -> str:
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename
    file_path.write_bytes(content)
    logger.info(f"Saved locally: {file_path}")
    return str(file_path)


def get_image_url(scan_id: str, image_path: str, url_type: str = "original") -> str:
    """
    Return a URL for serving an image.

    In cloud mode, image_path is already a public URL — return it directly.
    In local mode, return the API endpoint that serves the file.
    """
    if is_cloud_storage():
        return image_path  # already a public URL stored in DB
    # Local dev: serve via API endpoint
    suffix = "annotated" if url_type == "annotated" else "image"
    return f"/api/wounds/{scan_id}/{suffix}"
