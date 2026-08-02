"""Attachment normalization for CADAgent execution requests.

The add-in sends attachments as base64 payloads. Images keep the legacy sketch
vision path working. Text-like files are folded into prompt text so every BYOK
provider can use them. Binary documents are reported explicitly instead of
being silently ignored.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence
from xml.etree import ElementTree


TEXT_MIME_TYPES = {
    "application/json",
    "application/csv",
    "application/xml",
    "text/csv",
    "text/markdown",
    "text/plain",
    "text/tab-separated-values",
    "text/xml",
}
TEXT_EXTENSIONS = {".csv", ".json", ".md", ".markdown", ".txt", ".xml", ".tsv"}
DOCX_MIME_TYPES = {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
DOCX_EXTENSIONS = {".docx"}
IMAGE_MIME_PREFIX = "image/"
MAX_TEXT_ATTACHMENT_CHARS = 40_000
MAX_TOTAL_ATTACHMENT_CHARS = 100_000


@dataclass
class NormalizedAttachments:
    """Provider-safe representation of request attachments."""

    prompt_context: str = ""
    first_image_data: Optional[str] = None
    first_image_format: Optional[str] = None
    normalized: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def normalize_request_attachments(request: Mapping[str, Any]) -> NormalizedAttachments:
    """Normalize modern ``attachments[]`` plus legacy ``image_data`` fields."""

    attachments = _coerce_attachments(request.get("attachments"))
    legacy_image_data = request.get("image_data")
    if isinstance(legacy_image_data, str) and legacy_image_data.strip():
        if not _has_matching_image_attachment(attachments, legacy_image_data):
            image_format = str(request.get("image_format") or "png").strip().lower() or "png"
            attachments.insert(
                0,
                {
                    "name": "legacy_image",
                    "kind": "image",
                    "mime_type": _image_mime_from_format(image_format),
                    "format": image_format,
                    "data": legacy_image_data,
                    "source": "legacy_image_data",
                },
            )

    result = NormalizedAttachments()
    if not attachments:
        return result

    context_sections: List[str] = []
    total_chars = 0

    for index, attachment in enumerate(attachments, start=1):
        if not isinstance(attachment, Mapping):
            result.warnings.append(f"Attachment #{index} was ignored because it was not an object.")
            continue

        name = _safe_name(attachment.get("name"), fallback=f"attachment_{index}")
        mime_type = _safe_mime(attachment.get("mime_type") or attachment.get("type"))
        kind = _infer_kind(attachment, mime_type, name)
        data = attachment.get("data")
        size = attachment.get("size")

        normalized_item: Dict[str, Any] = {
            "name": name,
            "kind": kind,
            "mime_type": mime_type,
        }
        if isinstance(size, int):
            normalized_item["size"] = size

        if kind == "image":
            if isinstance(data, str) and data.strip():
                image_format = _image_format(attachment, mime_type)
                if result.first_image_data is None:
                    result.first_image_data = data
                    result.first_image_format = image_format
                normalized_item["format"] = image_format
                context_sections.append(
                    f"- {name}: image attachment ({mime_type or image_format}); "
                    "processed through the sketch vision path when image analysis is available."
                )
            else:
                warning = f"{name}: image attachment is missing base64 data."
                result.warnings.append(warning)
                context_sections.append(f"- {warning}")
            result.normalized.append(normalized_item)
            continue

        if kind == "text":
            decoded_text, warning = _decode_text_attachment(data)
            if warning:
                result.warnings.append(f"{name}: {warning}")
                context_sections.append(f"- {name}: {warning}")
            elif decoded_text is not None:
                remaining = max(0, MAX_TOTAL_ATTACHMENT_CHARS - total_chars)
                clipped = decoded_text[: min(MAX_TEXT_ATTACHMENT_CHARS, remaining)]
                total_chars += len(clipped)
                truncated = len(clipped) < len(decoded_text)
                normalized_item["chars"] = len(decoded_text)
                normalized_item["included_chars"] = len(clipped)
                context_sections.append(
                    _format_text_attachment_context(
                        name=name,
                        mime_type=mime_type,
                        text=clipped,
                        truncated=truncated,
                    )
                )
            result.normalized.append(normalized_item)
            continue

        if kind == "pdf":
            decoded_text, warning = _decode_pdf_attachment(data)
            if warning:
                result.warnings.append(f"{name}: {warning}")
                context_sections.append(f"- {name}: {warning}")
            elif decoded_text is not None:
                total_chars = _append_text_context(
                    context_sections,
                    normalized_item,
                    name=name,
                    mime_type=mime_type or "application/pdf",
                    text=decoded_text,
                    total_chars=total_chars,
                )
            result.normalized.append(normalized_item)
            continue

        if kind == "docx":
            decoded_text, warning = _decode_docx_attachment(data)
            if warning:
                result.warnings.append(f"{name}: {warning}")
                context_sections.append(f"- {name}: {warning}")
            elif decoded_text is not None:
                total_chars = _append_text_context(
                    context_sections,
                    normalized_item,
                    name=name,
                    mime_type=mime_type or "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    text=decoded_text,
                    total_chars=total_chars,
                )
            result.normalized.append(normalized_item)
            continue

        warning = (
            f"Unsupported attachment type ({mime_type or 'unknown'}). "
            "Supported provider-safe attachments are images, PDFs, DOCX files, and text/markdown/csv/json files."
        )
        result.warnings.append(f"{name}: {warning}")
        context_sections.append(f"- {name}: {warning}")
        result.normalized.append(normalized_item)

    if context_sections:
        result.prompt_context = "Attachments:\n" + "\n\n".join(context_sections)

    return result


def _coerce_attachments(raw: Any) -> List[Mapping[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [raw]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return [item for item in raw if isinstance(item, Mapping)]
    return []


def _has_matching_image_attachment(attachments: Sequence[Mapping[str, Any]], image_data: str) -> bool:
    for attachment in attachments:
        if not isinstance(attachment, Mapping):
            continue
        if attachment.get("data") != image_data:
            continue
        name = _safe_name(attachment.get("name"), fallback="attachment")
        mime_type = _safe_mime(attachment.get("mime_type") or attachment.get("type"))
        if _infer_kind(attachment, mime_type, name) == "image":
            return True
    return False


def _safe_name(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()[:160]
    return fallback


def _safe_mime(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()[:120]
    return ""


def _infer_kind(attachment: Mapping[str, Any], mime_type: str, name: str) -> str:
    raw_kind = attachment.get("kind")
    if isinstance(raw_kind, str) and raw_kind.strip().lower() in {"image", "text", "pdf", "docx", "document"}:
        kind = raw_kind.strip().lower()
        if kind == "document":
            if mime_type == "application/pdf":
                return "pdf"
            if mime_type in DOCX_MIME_TYPES:
                return "docx"
            return "unsupported"
        return kind

    lower_name = name.lower()
    extension = ""
    if "." in lower_name:
        extension = lower_name[lower_name.rfind(".") :]

    if mime_type.startswith(IMAGE_MIME_PREFIX):
        return "image"
    if mime_type == "application/pdf" or extension == ".pdf":
        return "pdf"
    if mime_type in DOCX_MIME_TYPES or extension in DOCX_EXTENSIONS:
        return "docx"
    if mime_type.startswith("text/") or mime_type in TEXT_MIME_TYPES or extension in TEXT_EXTENSIONS:
        return "text"
    return "unsupported"


def _decode_text_attachment(data: Any) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(data, str) or not data.strip():
        return None, "text attachment is missing base64 data."

    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None, "text attachment data is not valid base64."

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            text = ""
    else:  # pragma: no cover - latin-1 should always decode
        return None, "text attachment could not be decoded."

    text = text.replace("\x00", "")
    if not text.strip():
        return None, "text attachment was empty after decoding."
    return text, None


def _decode_pdf_attachment(data: Any) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(data, str) or not data.strip():
        return None, "PDF attachment is missing base64 data."

    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None, "PDF attachment data is not valid base64."

    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return (
            None,
            "PDF text extraction dependency is not available. Install pypdf to include PDF contents in the prompt.",
        )

    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for page_index, page in enumerate(reader.pages[:20], start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[page {page_index}]\n{text.strip()}")
    except Exception as exc:
        return None, f"PDF text extraction failed: {exc}"

    combined = "\n\n".join(pages).strip()
    if not combined:
        return None, "PDF text extraction found no selectable text."
    return combined, None


def _decode_docx_attachment(data: Any) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(data, str) or not data.strip():
        return None, "DOCX attachment is missing base64 data."

    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None, "DOCX attachment data is not valid base64."

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            xml_names = [
                name
                for name in archive.namelist()
                if name == "word/document.xml"
                or name.startswith("word/header")
                or name.startswith("word/footer")
                or name.startswith("word/footnotes")
                or name.startswith("word/endnotes")
            ]
            paragraphs: List[str] = []
            for xml_name in sorted(xml_names):
                paragraphs.extend(_extract_docx_xml_text(archive.read(xml_name)))
    except zipfile.BadZipFile:
        return None, "DOCX attachment is not a valid .docx file."
    except Exception as exc:
        return None, f"DOCX text extraction failed: {exc}"

    combined = "\n".join(paragraphs).strip()
    if not combined:
        return None, "DOCX text extraction found no readable text."
    return combined, None


def _extract_docx_xml_text(xml_bytes: bytes) -> List[str]:
    root = ElementTree.fromstring(xml_bytes)
    paragraphs: List[str] = []

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1] if "}" in tag else tag

    for paragraph in root.iter():
        if local_name(paragraph.tag) != "p":
            continue
        parts: List[str] = []
        for node in paragraph.iter():
            tag = local_name(node.tag)
            if tag == "t" and node.text:
                parts.append(node.text)
            elif tag == "tab":
                parts.append("\t")
            elif tag in {"br", "cr"}:
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _append_text_context(
    context_sections: List[str],
    normalized_item: Dict[str, Any],
    *,
    name: str,
    mime_type: str,
    text: str,
    total_chars: int,
) -> int:
    remaining = max(0, MAX_TOTAL_ATTACHMENT_CHARS - total_chars)
    clipped = text[: min(MAX_TEXT_ATTACHMENT_CHARS, remaining)]
    total_chars += len(clipped)
    truncated = len(clipped) < len(text)
    normalized_item["chars"] = len(text)
    normalized_item["included_chars"] = len(clipped)
    context_sections.append(
        _format_text_attachment_context(
            name=name,
            mime_type=mime_type,
            text=clipped,
            truncated=truncated,
        )
    )
    return total_chars


def _format_text_attachment_context(name: str, mime_type: str, text: str, truncated: bool) -> str:
    suffix = "\n[truncated]" if truncated else ""
    label = mime_type or "text"
    return f"- {name}: {label}\n```text\n{text}{suffix}\n```"


def _image_mime_from_format(image_format: str) -> str:
    normalized = image_format.lower().replace("jpg", "jpeg")
    if normalized not in {"png", "jpeg", "webp", "gif"}:
        normalized = "png"
    return f"image/{normalized}"


def _image_format(attachment: Mapping[str, Any], mime_type: str) -> str:
    raw_format = attachment.get("format")
    if isinstance(raw_format, str) and raw_format.strip():
        fmt = raw_format.strip().lower()
    elif mime_type.startswith("image/"):
        fmt = mime_type.split("/", 1)[1].lower()
    else:
        fmt = "png"
    if fmt == "jpg":
        fmt = "jpeg"
    return fmt


def attachment_debug_summary(normalized: NormalizedAttachments) -> Dict[str, Any]:
    """Small serializable summary suitable for logs/tests."""

    return {
        "count": len(normalized.normalized),
        "kinds": [item.get("kind") for item in normalized.normalized],
        "has_prompt_context": bool(normalized.prompt_context),
        "has_image": normalized.first_image_data is not None,
        "warnings": list(normalized.warnings),
    }
