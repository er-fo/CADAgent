import base64
import io
import zipfile

from backend.attachments import normalize_request_attachments


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _docx_b64(text: str) -> str:
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>"
        f"{text}"
        "</w:t></w:r></w:p></w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_text_attachment_is_added_to_prompt_context() -> None:
    result = normalize_request_attachments(
        {
            "attachments": [
                {
                    "name": "notes.md",
                    "kind": "text",
                    "mime_type": "text/markdown",
                    "data": _b64("# Mounting notes\nUse M3 screws."),
                    "size": 28,
                }
            ]
        }
    )

    assert result.first_image_data is None
    assert result.normalized[0]["kind"] == "text"
    assert "notes.md" in result.prompt_context
    assert "Use M3 screws." in result.prompt_context


def test_pdf_attachment_is_reported_when_text_extraction_is_unavailable() -> None:
    result = normalize_request_attachments(
        {
            "attachments": [
                {
                    "name": "drawing.pdf",
                    "kind": "pdf",
                    "mime_type": "application/pdf",
                    "data": base64.b64encode(b"%PDF").decode("ascii"),
                }
            ]
        }
    )

    assert result.normalized[0]["kind"] == "pdf"
    assert "drawing.pdf" in result.prompt_context
    assert "PDF text extraction" in result.prompt_context
    assert result.warnings


def test_docx_attachment_text_is_added_to_prompt_context() -> None:
    result = normalize_request_attachments(
        {
            "attachments": [
                {
                    "name": "brief.docx",
                    "kind": "docx",
                    "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "data": _docx_b64("Use four M4 screws on the mounting flange."),
                }
            ]
        }
    )

    assert result.normalized[0]["kind"] == "docx"
    assert "brief.docx" in result.prompt_context
    assert "Use four M4 screws" in result.prompt_context
    assert not result.warnings


def test_legacy_image_data_still_populates_first_image() -> None:
    result = normalize_request_attachments({"image_data": "abc123", "image_format": "jpg"})

    assert result.first_image_data == "abc123"
    assert result.first_image_format == "jpeg"
    assert result.normalized[0]["kind"] == "image"


def test_legacy_image_data_does_not_duplicate_matching_image_attachment() -> None:
    result = normalize_request_attachments(
        {
            "image_data": "abc123",
            "image_format": "png",
            "attachments": [
                {
                    "name": "sketch.png",
                    "kind": "image",
                    "mime_type": "image/png",
                    "format": "png",
                    "data": "abc123",
                }
            ],
        }
    )

    assert result.first_image_data == "abc123"
    assert [item["name"] for item in result.normalized] == ["sketch.png"]
