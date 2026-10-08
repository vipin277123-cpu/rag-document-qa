import re
import zipfile
from io import BytesIO
from pathlib import Path


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MIME_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".txt": {"text/plain", "application/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    },
}


def sanitize_filename(filename: str) -> str:
    """Keep a display-safe basename and remove path components and control characters."""
    basename = Path(filename.replace("\\", "/")).name
    suffix = Path(basename).suffix.lower()
    stem = Path(basename).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("._-")[:100]
    if not safe_stem:
        safe_stem = "document"
    return f"{safe_stem}{suffix}"


def validate_upload(filename: str, content_type: str | None, content: bytes) -> str:
    """Validate extension, practical MIME type, and basic file structure."""
    safe_name = sanitize_filename(filename)
    suffix = Path(safe_name).suffix
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload a PDF, TXT, or DOCX document.")

    normalized_type = (content_type or "application/octet-stream").split(";", 1)[0].strip().lower()
    if normalized_type not in MIME_TYPES[suffix]:
        raise ValueError(f"MIME type '{normalized_type}' does not match {suffix}.")

    if suffix == ".pdf" and not content.lstrip().startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF document.")
    if suffix == ".docx":
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                if "word/document.xml" not in archive.namelist():
                    raise ValueError("The uploaded DOCX file has no document content.")
        except zipfile.BadZipFile as error:
            raise ValueError("The uploaded file is not a valid DOCX document.") from error
    if suffix == ".txt":
        try:
            content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("TXT files must use UTF-8 encoding.") from error

    return safe_name
