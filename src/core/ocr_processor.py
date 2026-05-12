# src/core/ocr_processor.py
"""
OCR Pipeline sử dụng RapidOCR để nhận diện text từ ảnh trong tài liệu.
RapidOCR được chọn vì tính tương thích cao với Python 3.14+ thông qua ONNX Runtime.

Hỗ trợ:
- PDF digital (có text layer): tách ảnh nhúng (biểu đồ, con dấu) → OCR
- PDF scan (không có text layer): render toàn trang → OCR
- DOCX: tách ảnh nhúng → OCR

RapidOCR được lazy-initialized (singleton) để tránh load model nhiều lần.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

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
    Wrapper RapidOCR với lazy initialization và singleton pattern.

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
        """Lazy init RapidOCR engine."""
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                logger.info(f"Khởi tạo RapidOCR (lang={self._lang})...")
                # RapidOCR tự động detect và sử dụng các model phù hợp
                self._engine = RapidOCR()
                logger.info("RapidOCR đã sẵn sàng.")
            except ImportError:
                raise ImportError(
                    "RapidOCR chưa được cài đặt. Chạy: pip install rapidocr-onnxruntime"
                )
        return self._engine

    # -----------------------------------------------------------------------
    # OCR trên một ảnh (PIL Image hoặc numpy array)
    # -----------------------------------------------------------------------

    def run_ocr_on_image(self, image) -> tuple[str, float, list[str]]:
        """
        Chạy RapidOCR trên một ảnh.

        Args:
            image: PIL.Image hoặc numpy array

        Returns:
            (text, avg_confidence, lines_list)
            - text: toàn bộ text đã ghép, lọc theo min_confidence
            - avg_confidence: trung bình confidence
            - lines_list: danh sách từng dòng text
        """
        import numpy as np

        engine = self._get_ocr_engine()

        # Chuyển PIL Image → numpy array nếu cần
        if hasattr(image, "convert"):
            image = np.array(image.convert("RGB"))

        try:
            # RapidOCR trả về: [result, elapse]
            # result: [[box, text, confidence], ...]
            result, _ = engine(image)
        except Exception as e:
            logger.warning(f"RapidOCR gặp lỗi khi xử lý ảnh: {e}")
            return "", 0.0, []

        if not result:
            return "", 0.0, []

        lines = []
        confidences = []
        for line in result:
            if not line or len(line) < 3:
                continue
            text_content, conf = line[1], line[2]
            try:
                f_conf = float(conf)
            except (ValueError, TypeError):
                f_conf = 0.0
                
            if f_conf >= self._min_confidence and text_content.strip():
                lines.append(text_content.strip())
                confidences.append(f_conf)

        if not lines:
            return "", 0.0, []

        combined_text = "\n".join(lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return combined_text, avg_conf, lines

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

        Logic:
        - Nếu trang có ít hơn `scan_text_threshold` ký tự text → trang scan
          → render toàn trang thành ảnh → OCR
        - Nếu trang có text đủ → trang digital
          → chỉ tách các ảnh nhúng → OCR từng ảnh

        Args:
            file_path: đường dẫn đến file PDF
            dpi: độ phân giải khi render trang scan (cao hơn = chất lượng tốt hơn, chậm hơn)
            scan_text_threshold: số ký tự tối thiểu để coi trang là digital

        Returns:
            Danh sách OCRResult, có thể rỗng nếu không có ảnh/scan.
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
            page_number = page_num + 1  # 1-indexed

            # Lấy text có sẵn của trang
            existing_text = page.get_text("text").strip()
            is_scanned_page = len(existing_text) < scan_text_threshold

            if is_scanned_page:
                # ── Trang scan: render toàn trang → OCR ──────────────────
                logger.debug(f"  Trang {page_number}: scan (text={len(existing_text)} ký tự) → OCR full page")
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
                # ── Trang digital: tách ảnh nhúng → OCR ──────────────────
                embedded_results = self._ocr_embedded_images_in_page(page, page_number, path.name)
                if embedded_results:
                    logger.debug(f"  Trang {page_number}: digital → {len(embedded_results)} vùng ảnh OCR")
                results.extend(embedded_results)

        doc.close()
        logger.info(f"OCR hoàn tất '{path.name}': {len(results)} vùng ảnh được nhận diện")
        return results

    def _ocr_pdf_full_page(self, page, dpi: int = 200) -> tuple[str, float, list[str]]:
        """Render trang PDF thành ảnh rồi OCR."""
        try:
            import fitz
            mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 DPI là mặc định của PDF
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Chuyển sang numpy array
            import numpy as np
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            # PyMuPDF trả về RGB, đảm bảo 3 channels
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
                img_ext = base_image.get("ext", "png")

                image = Image.open(io.BytesIO(img_bytes))

                # Bỏ qua ảnh quá nhỏ (icon, watermark, v.v.)
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

        Args:
            file_path: đường dẫn đến file DOCX

        Returns:
            Danh sách OCRResult.
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

        # Lấy ảnh từ relationships của document
        image_parts = []
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    image_parts.append(rel.target_part)
                except Exception:
                    continue

        logger.info(f"OCR DOCX '{path.name}': {len(image_parts)} ảnh nhúng")

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
                        page_number=0,    # DOCX không có số trang rõ ràng
                        image_index=img_index,
                        confidence=avg_conf,
                        source_type="docx_image",
                        file_name=path.name,
                        ocr_lines=lines,
                    ))
            except Exception as e:
                logger.debug(f"Bỏ qua ảnh DOCX {img_index}: {e}")
                continue

        logger.info(f"OCR DOCX hoàn tất '{path.name}': {len(results)} vùng ảnh được nhận diện")
        return results


# ---------------------------------------------------------------------------
# Factory function (dùng từ bên ngoài)
# ---------------------------------------------------------------------------

def get_ocr_processor() -> OCRProcessor:
    """
    Trả về singleton OCRProcessor được cấu hình từ config.py.
    Import hàm này thay vì tạo OCRProcessor trực tiếp.
    """
    import config
    return OCRProcessor.get_instance(
        use_gpu=config.OCR_USE_GPU,
        lang=config.OCR_LANGUAGE,
        min_confidence=config.OCR_MIN_CONFIDENCE,
    )
