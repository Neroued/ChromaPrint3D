import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("OpenCV (cv2) is required for ColorCheckerTool.") from exc

Point = Tuple[float, float]


class ColorCheckerTool:
    """
    Build patch polygons from a reference LabelMe JSON, then transform them
    into a new image using four corner points of the ColorChecker chart.
    """

    def __init__(
        self,
        reference_json_path: Union[str, Path],
        label_order: Optional[Sequence[str]] = None,
        reference_desc_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.reference_json_path = Path(reference_json_path)
        self.ref_quad, self.patches = self._load_reference(self.reference_json_path)
        self.label_order = list(label_order) if label_order else None
        self.reference_colors = self._load_reference_colors(
            self.reference_json_path, reference_desc_path
        )
        self.reference_lab_d65 = {
            label: np.array(values["Lab_d65"], dtype=np.float32)
            for label, values in self.reference_colors.items()
            if "Lab_d65" in values
        }

    @staticmethod
    def _load_reference(path: Path) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        shapes = data.get("shapes", [])
        ref_poly = None
        patches: List[Dict[str, np.ndarray]] = []

        for shape in shapes:
            label = shape.get("label")
            points = shape.get("points") or []
            if not points:
                continue
            pts = np.array(points, dtype=np.float32)

            if label == "ColorChecker":
                ref_poly = pts
            else:
                patches.append({"label": label, "points": pts})

        if ref_poly is None:
            raise ValueError("Reference JSON must contain 'ColorChecker' polygon.")
        if ref_poly.shape[0] != 4:
            raise ValueError("Reference 'ColorChecker' polygon must have 4 points.")

        ref_quad = ColorCheckerTool._order_quad(ref_poly)
        return ref_quad, patches

    @staticmethod
    def _order_quad(pts: np.ndarray) -> np.ndarray:
        if pts.shape != (4, 2):
            pts = pts.reshape(4, 2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]       # top-left
        ordered[2] = pts[np.argmax(s)]       # bottom-right
        ordered[1] = pts[np.argmin(diff)]    # top-right
        ordered[3] = pts[np.argmax(diff)]    # bottom-left
        return ordered

    @staticmethod
    def _load_reference_colors(
        reference_json_path: Path,
        reference_desc_path: Optional[Union[str, Path]],
    ) -> Dict[str, Dict[str, List[float]]]:
        if reference_desc_path is not None:
            path = Path(reference_desc_path)
            if path.exists():
                return parse_colorchecker_desc(path)
            return {}

        candidates = [
            "/mnt/winshare/chroma/ColorChecker",
            reference_json_path.parent / "desc.txt",
            reference_json_path.parent.parent / "desc.txt",
        ]
        for candidate in candidates:
            path = Path(candidate)
            if path.is_dir():
                desc_path = path / "desc.txt"
                if desc_path.exists():
                    return parse_colorchecker_desc(desc_path)
                continue
            if path.exists():
                return parse_colorchecker_desc(path)
        return {}

    @staticmethod
    def _generate_candidate_quads(target_corners: Sequence[Point]) -> List[np.ndarray]:
        pts = np.array(target_corners, dtype=np.float32).reshape(4, 2)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        order = np.argsort(angles)
        cycle = pts[order]

        candidates: List[np.ndarray] = []
        for i in range(4):
            candidates.append(np.roll(cycle, -i, axis=0))

        reversed_cycle = cycle[::-1]
        for i in range(4):
            candidates.append(np.roll(reversed_cycle, -i, axis=0))
        return candidates

    @staticmethod
    def _shrink_polygon(pts: np.ndarray, ratio: float) -> np.ndarray:
        if ratio >= 1.0:
            return pts
        center = pts.mean(axis=0, keepdims=True)
        return center + (pts - center) * ratio

    @staticmethod
    def _load_image(image_or_path: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image_or_path, (str, Path)):
            image = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to read image: {image_or_path}")
            return image
        if isinstance(image_or_path, np.ndarray):
            return image_or_path
        raise TypeError("image_or_path must be a path or a numpy array.")

    @staticmethod
    def _sample_color(
        image: np.ndarray,
        polygon: np.ndarray,
        shrink_ratio: float = 0.85,
        method: str = "mean",
    ) -> Optional[np.ndarray]:
        if polygon.size == 0:
            return None
        poly = ColorCheckerTool._shrink_polygon(polygon, shrink_ratio)
        poly_int = np.round(poly).astype(np.int32)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly_int], 255)
        ys, xs = np.where(mask == 255)
        if xs.size == 0:
            return None
        pixels = image[ys, xs]

        if method == "median":
            return np.median(pixels, axis=0)
        return pixels.mean(axis=0)

    @staticmethod
    def _bgr_to_lab_d65(bgr: np.ndarray) -> List[float]:
        bgr_uint8 = np.clip(bgr, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        lab = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
        l = (lab[0] / 255.0) * 100.0
        a = lab[1] - 128.0
        b = lab[2] - 128.0
        return [round(float(l), 2), round(float(a), 2), round(float(b), 2)]

    @staticmethod
    def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
        return np.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4,
        )

    @staticmethod
    def _adapt_xyz_d65_to_d50(xyz_d65: np.ndarray) -> np.ndarray:
        bradford = np.array(
            [
                [0.8951, 0.2664, -0.1614],
                [-0.7502, 1.7135, 0.0367],
                [0.0389, -0.0685, 1.0296],
            ],
            dtype=np.float32,
        )
        bradford_inv = np.array(
            [
                [0.9869929, -0.1470543, 0.1599627],
                [0.4323053, 0.5183603, 0.0492912],
                [-0.0085287, 0.0400428, 0.9684867],
            ],
            dtype=np.float32,
        )
        white_d65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
        white_d50 = np.array([0.96422, 1.0, 0.82521], dtype=np.float32)

        lms_d65 = bradford @ white_d65
        lms_d50 = bradford @ white_d50
        scale = lms_d50 / lms_d65

        lms = bradford @ xyz_d65
        lms_adapted = lms * scale
        return bradford_inv @ lms_adapted

    @staticmethod
    def _xyz_to_lab(xyz: np.ndarray, white: np.ndarray) -> List[float]:
        delta = 6 / 29
        delta_cubed = delta ** 3
        scale = 1 / (3 * delta * delta)

        xyz_n = xyz / white
        f = np.where(
            xyz_n > delta_cubed,
            np.cbrt(xyz_n),
            xyz_n * scale + 4 / 29,
        )

        l = 116 * f[1] - 16
        a = 500 * (f[0] - f[1])
        b = 200 * (f[1] - f[2])
        return [round(float(l), 2), round(float(a), 2), round(float(b), 2)]

    @staticmethod
    def _bgr_to_lab_d50(bgr: np.ndarray) -> List[float]:
        rgb = np.clip(bgr, 0, 255)[::-1] / 255.0
        rgb_linear = ColorCheckerTool._srgb_to_linear(rgb)
        rgb_to_xyz = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
            dtype=np.float32,
        )
        xyz_d65 = rgb_to_xyz @ rgb_linear
        xyz_d50 = ColorCheckerTool._adapt_xyz_d65_to_d50(xyz_d65)
        white_d50 = np.array([0.96422, 1.0, 0.82521], dtype=np.float32)
        return ColorCheckerTool._xyz_to_lab(xyz_d50, white_d50)

    def transform_patches(
        self, target_corners: Sequence[Point]
    ) -> List[Dict[str, np.ndarray]]:
        target_quad = self._order_quad(np.array(target_corners, dtype=np.float32))
        h = cv2.getPerspectiveTransform(self.ref_quad, target_quad)
        return self._apply_homography(h)

    def _apply_homography(self, h: np.ndarray) -> List[Dict[str, np.ndarray]]:
        transformed: List[Dict[str, np.ndarray]] = []

        for patch in self.patches:
            pts = patch["points"].reshape(1, -1, 2).astype(np.float32)
            new_pts = cv2.perspectiveTransform(pts, h)[0]
            transformed.append({"label": patch["label"], "points": new_pts})
        return transformed

    def _orientation_error(
        self,
        image: np.ndarray,
        h: np.ndarray,
        shrink_ratio: float,
        method: str,
    ) -> float:
        errors: List[float] = []
        for patch in self.patches:
            ref_lab = self.reference_lab_d65.get(patch["label"])
            if ref_lab is None:
                continue
            pts = patch["points"].reshape(1, -1, 2).astype(np.float32)
            new_pts = cv2.perspectiveTransform(pts, h)[0]
            color_bgr = self._sample_color(
                image, new_pts, shrink_ratio=shrink_ratio, method=method
            )
            if color_bgr is None:
                continue
            lab = np.array(self._bgr_to_lab_d65(color_bgr), dtype=np.float32)
            errors.append(float(np.linalg.norm(lab - ref_lab)))

        if not errors:
            return float("inf")
        return float(np.mean(errors))

    def _select_homography(
        self,
        image: np.ndarray,
        target_corners: Sequence[Point],
        shrink_ratio: float,
        method: str,
    ) -> np.ndarray:
        if not self.reference_lab_d65:
            target_quad = self._order_quad(np.array(target_corners, dtype=np.float32))
            return cv2.getPerspectiveTransform(self.ref_quad, target_quad)

        best_h = None
        best_error = float("inf")
        for quad in self._generate_candidate_quads(target_corners):
            h = cv2.getPerspectiveTransform(self.ref_quad, quad)
            error = self._orientation_error(image, h, shrink_ratio, method)
            if error < best_error:
                best_error = error
                best_h = h

        if best_h is None:
            target_quad = self._order_quad(np.array(target_corners, dtype=np.float32))
            return cv2.getPerspectiveTransform(self.ref_quad, target_quad)
        return best_h

    def extract_patch_colors(
        self,
        image_or_path: Union[str, Path, np.ndarray],
        target_corners: Sequence[Point],
        shrink_ratio: float = 0.85,
        method: str = "mean",
    ) -> Dict[str, Dict[str, object]]:
        image = self._load_image(image_or_path)
        h = self._select_homography(image, target_corners, shrink_ratio, method)
        patches = self._apply_homography(h)

        ordered_labels = (
            self.label_order
            if self.label_order
            else [p["label"] for p in patches]
        )
        patch_map = {p["label"]: p for p in patches}

        results: Dict[str, Dict[str, object]] = {}
        for label in ordered_labels:
            patch = patch_map.get(label)
            if patch is None:
                continue
            color_bgr = self._sample_color(
                image, patch["points"], shrink_ratio=shrink_ratio, method=method
            )
            if color_bgr is None:
                continue

            color_rgb = color_bgr[::-1]
            entry: Dict[str, object] = {
                "sRGB": [int(round(v)) for v in color_rgb.tolist()],
                "Lab_d50": self._bgr_to_lab_d50(color_bgr),
                "Lab_d65": self._bgr_to_lab_d65(color_bgr),
            }

            results[label] = entry
        return results


def parse_colorchecker_desc(path: Union[str, Path]) -> Dict[str, Dict[str, List[float]]]:
    """
    Parse desc.txt into a dict mapping label -> color values.
    """
    text = Path(path).read_text(encoding="utf-8").strip().splitlines()
    result: Dict[str, Dict[str, List[float]]] = {}
    current_label: Optional[str] = None

    for line in text:
        line = line.strip()
        if not line:
            current_label = None
            continue
        if line[0].isdigit() and ". " in line:
            current_label = line.split(". ", 1)[1].rstrip(":")
            result[current_label] = {}
            continue
        if current_label and ":" in line:
            key, values = line.split(":", 1)
            values = values.strip().strip("[]")
            nums = [float(v.strip()) for v in values.split(",") if v.strip()]
            result[current_label][key] = nums

    return result

