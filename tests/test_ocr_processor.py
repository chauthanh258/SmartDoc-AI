# tests/test_ocr_processor.py
"""
Unit tests cho OCRProcessor.

Chạy: pytest tests/test_ocr_processor.py -v

Lưu ý: Một số test yêu cầu RapidOCR đã được cài đặt và file test thực tế.
       Test mock (không cần GPU/model) sẽ chạy đầu tiên.
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Thêm root project vào path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Test OCRResult dataclass
# ---------------------------------------------------------------------------

class TestOCRResult:
    def test_ocr_result_creation(self):
        from src.core.ocr_processor import OCRResult

        result = OCRResult(
            text="CÔNG TY ABC",
            page_number=2,
            image_index=0,
            confidence=0.95,
            source_type="embedded_image",
            file_name="test.pdf",
        )

        assert result.text == "CÔNG TY ABC"
        assert result.page_number == 2
        assert result.confidence == 0.95
        assert result.source_type == "embedded_image"
        assert result.ocr_lines == []  # default

    def test_ocr_result_with_lines(self):
        from src.core.ocr_processor import OCRResult

        result = OCRResult(
            text="line1\nline2",
            page_number=1,
            image_index=0,
            confidence=0.88,
            source_type="full_page_scan",
            file_name="scan.pdf",
            ocr_lines=["line1", "line2"],
        )

        assert len(result.ocr_lines) == 2


# ---------------------------------------------------------------------------
# Test OCRProcessor singleton
# ---------------------------------------------------------------------------

class TestOCRProcessorSingleton:
    def setup_method(self):
        from src.core.ocr_processor import OCRProcessor
        OCRProcessor.reset_instance()

    def test_singleton_returns_same_instance(self):
        from src.core.ocr_processor import OCRProcessor

        inst1 = OCRProcessor.get_instance()
        inst2 = OCRProcessor.get_instance()
        assert inst1 is inst2

    def test_reset_creates_new_instance(self):
        from src.core.ocr_processor import OCRProcessor

        inst1 = OCRProcessor.get_instance()
        OCRProcessor.reset_instance()
        inst2 = OCRProcessor.get_instance()
        assert inst1 is not inst2

    def test_get_ocr_processor_factory(self):
        from src.core.ocr_processor import get_ocr_processor, OCRProcessor

        with patch("config.OCR_USE_GPU", False), \
             patch("config.OCR_LANGUAGE", "vi"), \
             patch("config.OCR_MIN_CONFIDENCE", 0.6):
            processor = get_ocr_processor()
            assert isinstance(processor, OCRProcessor)


# ---------------------------------------------------------------------------
# Test run_ocr_on_image (mock PaddleOCR)
# ---------------------------------------------------------------------------

class TestRunOCROnImage:
    def setup_method(self):
        from src.core.ocr_processor import OCRProcessor
        OCRProcessor.reset_instance()

    def _make_mock_ocr_result(self, texts_confs):
        """Tạo output giả của RapidOCR."""
        lines = []
        for text, conf in texts_confs:
            # RapidOCR format: [[box, text, confidence], ...]
            lines.append([[[0, 0], [100, 0], [100, 30], [0, 30]], text, conf])
        return [lines, 0.1]  # [result, elapse]

    def test_run_ocr_returns_text(self):
        from src.core.ocr_processor import OCRProcessor
        import numpy as np

        processor = OCRProcessor.get_instance(min_confidence=0.5)

        mock_ocr_engine = MagicMock()
        mock_ocr_engine.return_value = self._make_mock_ocr_result([
            ("CÔNG TY ABC", 0.95),
            ("Đã xác nhận", 0.87),
        ])
        processor._engine = mock_ocr_engine

        dummy_image = np.zeros((100, 200, 3), dtype="uint8")
        text, conf, lines = processor.run_ocr_on_image(dummy_image)

        assert "CÔNG TY ABC" in text
        assert "Đã xác nhận" in text
        assert conf > 0.8
        assert len(lines) == 2

    def test_run_ocr_filters_low_confidence(self):
        from src.core.ocr_processor import OCRProcessor
        import numpy as np

        processor = OCRProcessor.get_instance(min_confidence=0.7)

        mock_ocr_engine = MagicMock()
        mock_ocr_engine.return_value = self._make_mock_ocr_result([
            ("Chữ rõ", 0.95),
            ("Chữ mờ", 0.50),  # Dưới ngưỡng → bị loại
        ])
        processor._engine = mock_ocr_engine

        dummy_image = MagicMock()
        dummy_image.convert.return_value = MagicMock()

        import numpy as np
        with patch("numpy.array", return_value=np.zeros((100, 100, 3), dtype="uint8")):
            text, conf, lines = processor.run_ocr_on_image(dummy_image)

        # Chỉ "Chữ rõ" vượt ngưỡng
        assert "Chữ rõ" in text
        assert len(lines) == 1

    def test_run_ocr_empty_image_returns_empty(self):
        from src.core.ocr_processor import OCRProcessor
        import numpy as np

        processor = OCRProcessor.get_instance()

        mock_ocr_engine = MagicMock()
        mock_ocr_engine.return_value = [[], 0.1]  # Không có text
        processor._engine = mock_ocr_engine

        dummy_image = np.zeros((50, 50, 3), dtype="uint8")
        text, conf, lines = processor.run_ocr_on_image(dummy_image)

        assert text == ""
        assert conf == 0.0
        assert lines == []

    def test_run_ocr_handles_string_confidence(self):
        from src.core.ocr_processor import OCRProcessor
        import numpy as np

        processor = OCRProcessor.get_instance(min_confidence=0.5)

        mock_ocr_engine = MagicMock()
        # Giả lập RapidOCR trả về confidence dạng string
        mock_ocr_engine.return_value = [
            [
                [[[0, 0], [100, 0], [100, 30], [0, 30]], "Text with string conf", "0.95"]
            ],
            0.1
        ]
        processor._engine = mock_ocr_engine

        dummy_image = np.zeros((50, 50, 3), dtype="uint8")
        text, conf, lines = processor.run_ocr_on_image(dummy_image)

        assert text == "Text with string conf"
        assert conf == 0.95



# ---------------------------------------------------------------------------
# Test process_pdf_for_ocr (mock fitz/PyMuPDF)
# ---------------------------------------------------------------------------

class TestProcessPDFForOCR:
    def setup_method(self):
        from src.core.ocr_processor import OCRProcessor
        OCRProcessor.reset_instance()

    def test_returns_empty_on_nonexistent_file(self):
        from src.core.ocr_processor import OCRProcessor

        processor = OCRProcessor.get_instance()
        # Không có fitz mock → nếu file không tồn tại, fitz.open() sẽ raise
        results = processor.process_pdf_for_ocr("/nonexistent/file.pdf")
        assert results == []

    def test_pdf_scan_page_detected(self):
        """Test logic phát hiện trang scan (< threshold ký tự)."""
        from src.core.ocr_processor import OCRProcessor
        import numpy as np

        processor = OCRProcessor.get_instance()
        processor._engine = MagicMock()
        processor._engine.return_value = [
            [  # Một dòng text
                [[[0, 0], [100, 0], [100, 30], [0, 30]], "Biên bản họp", 0.91]
            ],
            0.1
        ]

        mock_page = MagicMock()
        mock_page.get_text.return_value = "  "  # Trang scan (gần rỗng)
        mock_page.get_images.return_value = []

        # Mock pixmap
        pix = MagicMock()
        pix.samples = np.zeros(100 * 100 * 3, dtype="uint8").tobytes()
        pix.height = 100
        pix.width = 100
        pix.n = 3
        mock_page.get_pixmap.return_value = pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("fitz.open", return_value=mock_doc):
            results = processor.process_pdf_for_ocr(
                "test.pdf",
                dpi=72,
                scan_text_threshold=10,
            )

        # Trang scan phải tạo OCRResult với source_type='full_page_scan'
        assert len(results) >= 1
        assert results[0].source_type == "full_page_scan"
        assert "Biên bản họp" in results[0].text


# ---------------------------------------------------------------------------
# Test _apply_ocr_to_pdf (integration với document_loader)
# ---------------------------------------------------------------------------

class TestApplyOCRToPDF:
    def test_ocr_text_merged_into_existing_page(self):
        from langchain_core.documents import Document
        from src.core.document_loader import _apply_ocr_to_pdf
        from src.core.ocr_processor import OCRResult

        docs = [
            Document(
                page_content="Điều khoản hợp đồng...",
                metadata={"page_number": 1, "file_name": "hop_dong.pdf"},
            )
        ]

        mock_result = OCRResult(
            text="CÔNG TY ABC | ĐÃ XÁC NHẬN",
            page_number=1,
            image_index=0,
            confidence=0.92,
            source_type="embedded_image",
            file_name="hop_dong.pdf",
        )

        mock_config = MagicMock()
        mock_config.OCR_DPI = 200
        mock_config.OCR_SCAN_TEXT_THRESHOLD = 10

        with patch("src.core.document_loader.get_ocr_processor") as mock_get:
            mock_processor = MagicMock()
            mock_processor.process_pdf_for_ocr.return_value = [mock_result]
            mock_get.return_value = mock_processor

            result_docs = _apply_ocr_to_pdf(
                Path("hop_dong.pdf"), docs, mock_config
            )

        assert len(result_docs) == 1
        assert "Điều khoản hợp đồng" in result_docs[0].page_content
        assert "CÔNG TY ABC" in result_docs[0].page_content
        assert result_docs[0].metadata.get("has_ocr") is True

    def test_scan_page_text_replaced(self):
        """Trang scan với text rỗng: OCR text phải thay thế page_content."""
        from langchain_core.documents import Document
        from src.core.document_loader import _apply_ocr_to_pdf
        from src.core.ocr_processor import OCRResult

        docs = [
            Document(
                page_content="",   # Trang scan rỗng
                metadata={"page_number": 2, "file_name": "scan.pdf"},
            )
        ]

        mock_result = OCRResult(
            text="Nội dung scan trang 2",
            page_number=2,
            image_index=0,
            confidence=0.88,
            source_type="full_page_scan",
            file_name="scan.pdf",
        )

        mock_config = MagicMock()
        mock_config.OCR_DPI = 200
        mock_config.OCR_SCAN_TEXT_THRESHOLD = 10

        with patch("src.core.document_loader.get_ocr_processor") as mock_get:
            mock_processor = MagicMock()
            mock_processor.process_pdf_for_ocr.return_value = [mock_result]
            mock_get.return_value = mock_processor

            result_docs = _apply_ocr_to_pdf(
                Path("scan.pdf"), docs, mock_config
            )

        assert result_docs[0].page_content == "Nội dung scan trang 2"
        assert result_docs[0].metadata["has_ocr"] is True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
