#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import ChromaPrint3D and convert an image to 3MF."
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--db", required=True, help="Path to ColorDB json")
    parser.add_argument("--out", default=None, help="Output .3mf path")
    parser.add_argument(
        "--preview",
        default=None,
        help="Preview image path (default: <out>_preview.png)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="ImgProc request scale"
    )
    parser.add_argument(
        "--max-width", type=int, default=512, help="Max width (0 = no limit)"
    )
    parser.add_argument(
        "--max-height", type=int, default=512, help="Max height (0 = no limit)"
    )
    parser.add_argument(
        "--color-space",
        choices=["lab", "rgb"],
        default="lab",
        help="Match in Lab or RGB",
    )
    parser.add_argument("--k", type=int, default=1, help="Top-k candidates")
    parser.add_argument(
        "--flip-y", type=int, choices=[0, 1], default=1, help="Flip Y axis"
    )
    parser.add_argument("--pixel-mm", type=float, default=0.0, help="Pixel size in mm")
    parser.add_argument(
        "--layer-mm", type=float, default=0.0, help="Layer height in mm"
    )
    return parser.parse_args()


def ensure_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def default_out_path(image_path: str) -> str:
    p = Path(image_path)
    stem = p.stem if p.stem else "output"
    return str(p.with_name(f"{stem}.3mf"))


def default_preview_path(out_path: str) -> str:
    p = Path(out_path)
    stem = p.stem if p.stem else "preview"
    return str(p.with_name(f"{stem}_preview.png"))


def write_ppm(path: str, rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("PPM writer expects HxWx3 RGB image")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(rgb.tobytes())


def save_preview_image(bgr: np.ndarray, path: str) -> str:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Preview expects HxWx3 BGR image")
    rgb = bgr[:, :, ::-1].copy()
    suffix = Path(path).suffix.lower()
    if suffix in {".ppm", ".pnm"}:
        write_ppm(path, rgb)
        return path
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
        return path
    except Exception:
        pass
    try:
        import imageio.v2 as imageio  # type: ignore[import-not-found]

        imageio.imwrite(path, rgb)
        return path
    except Exception:
        pass
    fallback = str(Path(path).with_suffix(".ppm"))
    write_ppm(fallback, rgb)
    print(f"Preview saved as PPM (install pillow/imageio for PNG): {fallback}")
    return fallback


def main() -> int:
    args = parse_args()
    ensure_file(args.image, "Image")
    ensure_file(args.db, "ColorDB")

    out_path = args.out or default_out_path(args.image)
    preview_path = args.preview or default_preview_path(out_path)

    import ChromaPrint3D as c

    db = c.ColorDB.load_from_json(args.db)

    imgproc = c.ImgProc()
    if args.scale <= 0:
        raise ValueError("--scale must be positive")
    if args.max_width < 0 or args.max_height < 0:
        raise ValueError("--max-width/--max-height must be >= 0")
    imgproc.request_scale = args.scale
    imgproc.max_width = args.max_width
    imgproc.max_height = args.max_height

    img = imgproc.run(args.image)

    match_cfg = c.MatchConfig()
    match_cfg.k_candidates = args.k
    match_cfg.color_space = (
        c.ColorSpace.Rgb if args.color_space == "rgb" else c.ColorSpace.Lab
    )

    recipe_map = c.RecipeMap.MatchFromImage(img, db, match_cfg)

    preview_bgr = recipe_map.to_bgr_image(255, 255, 255)
    if preview_bgr is None or preview_bgr.size == 0:
        raise RuntimeError("Failed to generate preview image")
    save_preview_image(preview_bgr, preview_path)

    build_cfg = c.BuildModelIRConfig()
    build_cfg.flip_y = bool(args.flip_y)
    model = c.ModelIR.Build(recipe_map, db, build_cfg)

    mesh_cfg = c.BuildMeshConfig()
    if args.pixel_mm > 0.0:
        mesh_cfg.pixel_mm = args.pixel_mm
    else:
        mesh_cfg.pixel_mm = db.line_width_mm if db.line_width_mm > 0.0 else 1.0
    if args.layer_mm > 0.0:
        mesh_cfg.layer_height_mm = args.layer_mm
    else:
        mesh_cfg.layer_height_mm = (
            db.layer_height_mm if db.layer_height_mm > 0.0 else 0.08
        )

    c.Export3mf(out_path, model, mesh_cfg)
    print(f"Saved 3MF to {out_path}")
    print(f"Saved preview to {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
