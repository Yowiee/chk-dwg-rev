from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pymupdf
from rapidocr_onnxruntime import RapidOCR

RENDER_DPI = 300
BOTTOM_RIGHT_X = 0.55
BOTTOM_RIGHT_Y = 0.60

_ocr: RapidOCR | None = None


def _get_ocr() -> RapidOCR:
    global _ocr
    if _ocr is None:
        _ocr = RapidOCR()
    return _ocr


def render_first_page(pdf_path: Path) -> np.ndarray:
    """Render the first page of a PDF to an RGB numpy image."""
    with pymupdf.open(pdf_path) as doc:
        if doc.page_count == 0:
            raise ValueError("PDF has no pages")
        pix = doc[0].get_pixmap(dpi=RENDER_DPI)

    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        image = image[:, :, :3]
    return np.ascontiguousarray(image)


def is_engineering_drawing(image: np.ndarray) -> bool:
    """Use simple layout statistics to decide if an image looks like a drawing."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    white_ratio = float((gray >= 200).mean())

    binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(20, width // 40), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(20, height // 40))
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    line_ratio = (
        cv2.countNonZero(horizontal) + cv2.countNonZero(vertical)
    ) / float(width * height)
    horizontal_count = cv2.connectedComponents(horizontal)[0] - 1
    vertical_count = cv2.connectedComponents(vertical)[0] - 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    colour_ratio = float((hsv[:, :, 1] > 60).mean())

    return (
        white_ratio >= 0.55
        and line_ratio >= 0.005
        and horizontal_count >= 5
        and vertical_count >= 5
        and colour_ratio <= 0.65
    )


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def _is_rev_label(text: str) -> bool:
    normalized = _normalize(text)
    if normalized == "REV":
        return True
    if len(normalized) == 3 and normalized[0] == "R" and normalized[1] == "E":
        return sum(a != b for a, b in zip(normalized, "REV")) <= 1
    return False


def _clean_value(text: str) -> str:
    return "".join(
        ch for ch in text.upper() if ch.isalnum() or ch in "-./"
    )


def _recognize_region(region: np.ndarray) -> tuple[str, float]:
    if region.size == 0:
        return "", 0.0

    enlarged = cv2.resize(region, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    result, _ = _get_ocr()(
        cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB),
        use_det=False,
        use_cls=False,
        use_rec=True,
    )
    if not result:
        return "", 0.0
    text, score = result[0][0], float(result[0][1])
    return text, score


def _find_rev_labels(crop: np.ndarray) -> list[tuple[float, float, float, float]]:
    result, _ = _get_ocr()(crop)
    labels: list[tuple[float, float, float, float]] = []
    for box, text, _score in result or []:
        if not _is_rev_label(text):
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        labels.append((min(xs), min(ys), max(xs), max(ys)))
    return labels


def find_revision(image: np.ndarray) -> str | None:
    """Locate the REV title-block cell and return the characters beside / below it."""
    height, width = image.shape[:2]
    crop = image[int(height * BOTTOM_RIGHT_Y) :, int(width * BOTTOM_RIGHT_X) :]

    labels = _find_rev_labels(crop)
    if not labels:
        return None

    # The title-block REV is normally the lowest / right-most match.
    x0, y0, x1, y1 = max(labels, key=lambda box: (box[3], box[2]))
    label_width = max(1.0, x1 - x0)
    label_height = max(1.0, y1 - y0)

    # Primary attempt: the value sits directly below the REV label.
    vx0 = max(0, int(x0 - 0.27 * label_height))
    vx1 = min(crop.shape[1], int(x1 + 0.27 * label_height))
    vy0 = max(0, int(y1 - 0.13 * label_height))
    vy1 = min(crop.shape[0], int(y1 + 2.33 * label_height))
    text, score = _recognize_region(crop[vy0:vy1, vx0:vx1])
    value = _clean_value(text)
    if value and not _is_rev_label(value) and score >= 0.35:
        return value

    # Fallback: detect text below the label and choose the closest token.
    vx0 = max(0, int(x0 - 1.2 * label_width))
    vx1 = min(crop.shape[1], int(x1 + 1.2 * label_width))
    vy0 = max(0, int(y1 - 0.1 * label_height))
    vy1 = min(crop.shape[0], int(y1 + 2.5 * label_height))
    result, _ = _get_ocr()(crop[vy0:vy1, vx0:vx1])

    best: tuple[float, str] | None = None
    label_cx = (x0 + x1) / 2
    value_cy = y1 + 0.7 * label_height
    for box, text, score in result or []:
        if _is_rev_label(text) or float(score) < 0.5:
            continue
        candidate = _clean_value(text)
        if not candidate:
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        cx = (min(xs) + max(xs)) / 2 + vx0
        cy = (min(ys) + max(ys)) / 2 + vy0
        distance = ((cx - label_cx) ** 2 + (cy - value_cy) ** 2) ** 0.5
        if best is None or distance < best[0]:
            best = (distance, candidate)
    if best is not None:
        return best[1]

    # Last fallback: some title blocks print the value to the right of REV.
    vx0 = min(crop.shape[1], int(x1 + 0.15 * label_width))
    vx1 = min(crop.shape[1], int(x1 + 3.0 * label_width))
    vy0 = max(0, int(y0 - 0.3 * label_height))
    vy1 = min(crop.shape[0], int(y1 + 0.3 * label_height))
    text, score = _recognize_region(crop[vy0:vy1, vx0:vx1])
    value = _clean_value(text)
    if value and not _is_rev_label(value) and score >= 0.35:
        return value

    return None


def check_pdf(pdf_path: Path) -> tuple[bool, str | None]:
    """Return (is_engineering_drawing, revision) for the first page of a PDF."""
    image = render_first_page(pdf_path)
    if not is_engineering_drawing(image):
        return False, None
    return True, find_revision(image)
