# src/core/ocr_processor.py
"""
OCR Pipeline sử dụng PaddleOCR để nhận diện text từ ảnh trong tài liệu.
PaddleOCR được chọn vì hiệu suất vượt trội cho tiếng Việt và khả năng xử lý layout tốt.

Hỗ trợ:
- PDF digital (có text layer): tách ảnh nhúng (biểu đồ, con dấu) → OCR
- PDF scan (không có text layer): render toàn trang → OCR
- DOCX: tách ảnh nhúng → OCR

PaddleOCR được lazy-initialized (singleton) để tránh load model nhiều lần.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logger import logger

# ---------------------------------------------------------------------------
# OCRResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    """Kết quả OCR cho một vùng ảnh trong tài liệu."""

    text: str
    """Text trích xuất từ OCR (đã được làm sạch)."""

    page_number: int
    """Số trang (1-indexed). Bằng 0 nếu không xác định được (DOCX)."""

    image_index: int
    """Thứ tự ảnh trong trang (bắt đầu từ 0)."""

    confidence: float
    """Trung bình confidence của tất cả dòng text OCR (0.0 - 1.0)."""

    source_type: str
    """
    Loại nguồn ảnh:
    - 'embedded_image': ảnh nhúng trong trang PDF có text
    - 'full_page_scan':  trang PDF không có text (toàn ảnh)
    - 'docx_image':      ảnh nhúng trong DOCX
    """

    file_name: str
    """Tên file gốc chứa ảnh."""

    ocr_lines: list[str] = field(default_factory=list)
    """Danh sách từng dòng text OCR (để debug/display)."""


# ---------------------------------------------------------------------------
# Singleton OCRProcessor
# ---------------------------------------------------------------------------

class OCRProcessor:
    """
    Wrapper PaddleOCR với lazy initialization và singleton pattern.

    Sử dụng:
        processor = OCRProcessor.get_instance()
        results = processor.process_pdf_for_ocr("path/to/file.pdf")
    """

    _instance: OCRProcessor | None = None

    def __init__(self, use_gpu: bool = False, lang: str = "vi", min_confidence: float = 0.6):
        self._use_gpu = use_gpu
        self._lang = lang
        self._min_confidence = min_confidence
        self._engine = None  # lazy init khi cần
        
        # Mapping lang code cho PaddleOCR
        self._lang_map = {
            "vi": "vi",
            "en": "en",
            "ch": "ch"
        }

    @classmethod
    def get_instance(cls, use_gpu: bool = False, lang: str = "vi", min_confidence: float = 0.6) -> OCRProcessor:
        """Trả về singleton instance, tạo mới nếu chưa có."""
        if cls._instance is None:
            cls._instance = cls(use_gpu=use_gpu, lang=lang, min_confidence=min_confidence)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (dùng cho testing)."""
        cls._instance = None

    def _get_ocr_engine(self):
        """Lazy init RapidOCR engine (Lightweight & Stable)."""
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                logger.info("Khởi tạo RapidOCR...")
                self._engine = RapidOCR()
                self._engine_type = "rapid"
                logger.info("RapidOCR đã sẵn sàng.")
                return self._engine
            except ImportError:
                logger.error("Không tìm thấy RapidOCR. Vui lòng cài đặt: pip install rapidocr-onnxruntime")
                raise ImportError("RapidOCR chưa được cài đặt.")
        return self._engine

    # -----------------------------------------------------------------------
    # OCR trên một ảnh (PIL Image hoặc numpy array)
    # -----------------------------------------------------------------------

    def run_ocr_on_image(self, image) -> tuple[str, float, list[str]]:
        """
        Chạy OCR engine (RapidOCR).
        """
        import numpy as np
        engine = self._get_ocr_engine()
        
        if hasattr(image, "convert"):
            image = np.array(image.convert("RGB"))

        try:
            # RapidOCR output format: [ [ [box], text, confidence ], ... ]
            result, _ = engine(image)
            if not result: return "", 0.0, []
            
            lines, confs = [], []
            for line in result:
                if line and len(line) >= 3:
                    text_content = line[1].strip()
                    conf = float(line[2])
                    if text_content and conf >= self._min_confidence:
                        lines.append(text_content)
                        confs.append(conf)
            
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            return "\n".join(lines), avg_conf, lines
                
        except Exception as e:
            logger.warning(f"OCR gặp lỗi: {e}")
            return "", 0.0, []

    # -----------------------------------------------------------------------
    # Tách ảnh từ PDF và chạy OCR
    # -----------------------------------------------------------------------

    def process_pdf_for_ocr(
        self,
        file_path: str | Path,
        dpi: int = 200,
        scan_text_threshold: int = 10,
    ) -> list[OCRResult]:
        """
        Xử lý OCR cho toàn bộ file PDF.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF chưa được cài đặt. Chạy: pip install pymupdf")

        path = Path(file_path)
        results: list[OCRResult] = []

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            logger.error(f"Không thể mở file PDF {path.name}: {e}")
            return []

        logger.info(f"OCR PDF '{path.name}': {len(doc)} trang")

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_number = page_num + 1

            existing_text = page.get_text("text").strip()
            is_scanned_page = len(existing_text) < scan_text_threshold

            if is_scanned_page:
                logger.debug(f"  Trang {page_number}: scan → OCR full page")
                ocr_text, avg_conf, lines = self._ocr_pdf_full_page(page, dpi=dpi)

                if ocr_text.strip():
                    results.append(OCRResult(
                        text=ocr_text,
                        page_number=page_number,
                        image_index=0,
                        confidence=avg_conf,
                        source_type="full_page_scan",
                        file_name=path.name,
                        ocr_lines=lines,
                    ))
            else:
                embedded_results = self._ocr_embedded_images_in_page(page, page_number, path.name)
                if embedded_results:
                    logger.debug(f"  Trang {page_number}: digital → {len(embedded_results)} ảnh nhúng OCR")
                results.extend(embedded_results)

        doc.close()
        return results

    def _ocr_pdf_full_page(self, page, dpi: int = 200) -> tuple[str, float, list[str]]:
        """Render trang PDF thành ảnh rồi OCR."""
        try:
            import fitz
            import numpy as np
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                img_array = img_array[:, :, :3]

            return self.run_ocr_on_image(img_array)
        except Exception as e:
            logger.warning(f"Không thể render trang PDF: {e}")
            return "", 0.0, []

    def _ocr_embedded_images_in_page(
        self, page, page_number: int, file_name: str
    ) -> list[OCRResult]:
        """Tách và OCR tất cả ảnh nhúng trong một trang PDF."""
        from PIL import Image
        import fitz

        results: list[OCRResult] = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
                img_bytes = base_image["image"]
                image = Image.open(io.BytesIO(img_bytes))

                if image.width < 50 or image.height < 50:
                    continue

                ocr_text, avg_conf, lines = self.run_ocr_on_image(image)

                if ocr_text.strip():
                    results.append(OCRResult(
                        text=ocr_text,
                        page_number=page_number,
                        image_index=img_index,
                        confidence=avg_conf,
                        source_type="embedded_image",
                        file_name=file_name,
                        ocr_lines=lines,
                    ))
            except Exception as e:
                logger.debug(f"Bỏ qua ảnh {img_index} trang {page_number}: {e}")
                continue

        return results

    # -----------------------------------------------------------------------
    # Tách ảnh từ DOCX và chạy OCR
    # -----------------------------------------------------------------------

    def process_docx_for_ocr(self, file_path: str | Path) -> list[OCRResult]:
        """
        Xử lý OCR cho tất cả ảnh nhúng trong file DOCX/DOCM.
        """
        try:
            from docx import Document as DocxDocument
            from PIL import Image
        except ImportError:
            raise ImportError("python-docx hoặc Pillow chưa được cài đặt.")

        path = Path(file_path)
        results: list[OCRResult] = []

        try:
            doc = DocxDocument(str(path))
        except Exception as e:
            logger.error(f"Không thể mở DOCX {path.name}: {e}")
            return []

        image_parts = []
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    image_parts.append(rel.target_part)
                except Exception:
                    continue

        for img_index, img_part in enumerate(image_parts):
            try:
                img_bytes = img_part.blob
                image = Image.open(io.BytesIO(img_bytes))

                if image.width < 50 or image.height < 50:
                    continue

                ocr_text, avg_conf, lines = self.run_ocr_on_image(image)

                if ocr_text.strip():
                    results.append(OCRResult(
                        text=ocr_text,
                        page_number=0,
                        image_index=img_index,
                        confidence=avg_conf,
                        source_type="docx_image",
                        file_name=path.name,
                        ocr_lines=lines,
                    ))
            except Exception as e:
                logger.debug(f"Bỏ qua ảnh DOCX {img_index}: {e}")
                continue

        return results


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_ocr_processor() -> OCRProcessor:
    """
    Trả về singleton OCRProcessor được cấu hình từ config.py.
    """
    import config
    return OCRProcessor.get_instance(
        use_gpu=config.OCR_USE_GPU,
        lang=config.OCR_LANGUAGE,
        min_confidence=config.OCR_MIN_CONFIDENCE,
    )

