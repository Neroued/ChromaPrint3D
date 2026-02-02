"""ChromaPrint3D core bindings."""

from __future__ import annotations

from typing import overload
import enum
import numpy as np


class ResizeMethod(enum.IntEnum):
    """Resize sampling method."""

    Nearest: ResizeMethod
    Area: ResizeMethod
    Linear: ResizeMethod
    Cubic: ResizeMethod


class DenoiseMethod(enum.IntEnum):
    """Denoising algorithm."""

    None: DenoiseMethod
    Bilateral: DenoiseMethod
    Median: DenoiseMethod


class LayerOrder(enum.IntEnum):
    """Stacking order of layers."""

    Top2Bottom: LayerOrder
    Bottom2Top: LayerOrder


class ColorSpace(enum.IntEnum):
    """Color space selection."""

    Lab: ColorSpace
    Rgb: ColorSpace


class Vec3i:
    """3D vector with integer components."""

    x: int
    y: int
    z: int

    def __init__(self, x: int = 0, y: int = 0, z: int = 0) -> None: ...
    def dot(self, other: Vec3i) -> int: ...
    def length_squared(self) -> int: ...


class Vec3f:
    """3D vector with float components."""

    x: float
    y: float
    z: float

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None: ...
    def dot(self, other: Vec3f) -> float: ...
    def length_squared(self) -> float: ...
    def length(self) -> float: ...
    def normalized(self) -> Vec3f: ...
    def is_finite(self) -> bool: ...
    def nearly_equal(self, other: Vec3f, eps: float = 1e-5) -> bool: ...


class Rgb(Vec3f):
    """RGB color in linear space."""

    r: float
    g: float
    b: float

    def __init__(self, r: float = 0.0, g: float = 0.0, b: float = 0.0) -> None: ...
    def to_lab(self) -> Lab: ...
    def to_rgb255(self) -> tuple[int, int, int]: ...
    @staticmethod
    def from_lab(lab: Lab) -> Rgb: ...
    @staticmethod
    def from_rgb255(r: int, g: int, b: int) -> Rgb: ...


class Lab(Vec3f):
    """CIELAB color."""

    l: float
    a: float
    b: float

    def __init__(self, l: float = 0.0, a: float = 0.0, b: float = 0.0) -> None: ...
    def to_rgb(self) -> Rgb: ...
    @staticmethod
    def from_rgb(rgb: Rgb) -> Lab: ...
    @staticmethod
    def delta_e76(lab1: Lab, lab2: Lab) -> float: ...


class Channel:
    """Channel metadata."""

    color: str
    material: str

    def __init__(self) -> None: ...


class Entry:
    """A palette entry with Lab color and recipe."""

    lab: Lab
    recipe: list[int]

    def __init__(self) -> None: ...
    def color_layers(self) -> int: ...


class ColorDB:
    """Color database for mapping colors to recipes."""

    name: str
    max_color_layers: int
    base_layers: int
    base_channel_idx: int
    layer_height_mm: float
    line_width_mm: float
    layer_order: LayerOrder
    palette: list[Channel]
    entries: list[Entry]

    def __init__(self) -> None: ...
    def num_channels(self) -> int: ...
    @staticmethod
    def load_from_json(path: str) -> ColorDB: ...
    def save_to_json(self, path: str) -> None: ...
    def nearest_entry_lab(self, target: Lab) -> Entry: ...
    def nearest_entry_rgb(self, target: Rgb) -> Entry: ...
    def nearest_entries_lab(self, target: Lab, k: int) -> list[Entry]: ...
    def nearest_entries_rgb(self, target: Rgb, k: int) -> list[Entry]: ...


class ImgProcResult:
    """Image processing output."""

    name: str
    width: int
    height: int
    rgb: np.ndarray
    lab: np.ndarray
    mask: np.ndarray


class ImgProc:
    """Image preprocessing pipeline."""

    request_scale: float
    max_width: int
    max_height: int
    use_alpha_mask: bool
    alpha_threshold: int
    upsample_method: ResizeMethod
    downsample_method: ResizeMethod
    denoise_method: DenoiseMethod
    denoise_kernel: int
    bilateral_diameter: int
    bilateral_sigma_color: float
    bilateral_sigma_space: float

    def __init__(self) -> None: ...
    def run(self, path: str) -> ImgProcResult: ...


class MatchConfig:
    """Matching configuration."""

    k_candidates: int
    color_space: ColorSpace

    def __init__(self) -> None: ...


class RecipeMap:
    """Matched recipe map."""

    name: str
    width: int
    height: int
    color_layers: int
    num_channels: int
    layer_order: LayerOrder
    recipes: list[int]
    mask: list[int]
    mapped_color: list[Lab]

    def __init__(self) -> None: ...
    def recipe_at(self, row: int, col: int) -> list[int]: ...
    def mask_at(self, row: int, col: int) -> int: ...
    def color_at(self, row: int, col: int) -> Lab: ...
    def to_bgr_image(self, background_b: int = 0, background_g: int = 0,
                     background_r: int = 0) -> np.ndarray: ...
    @staticmethod
    def MatchFromImage(img: ImgProcResult, db: ColorDB, cfg: MatchConfig = MatchConfig()) -> RecipeMap: ...


class VoxelGrid:
    """Voxel occupancy grid."""

    width: int
    height: int
    num_layers: int
    channel_idx: int
    ooc: list[int]

    def __init__(self) -> None: ...
    def get(self, w: int, h: int, l: int) -> bool: ...
    def set(self, w: int, h: int, l: int, value: bool) -> bool: ...


class BuildModelIRConfig:
    """ModelIR build options."""

    flip_y: bool
    base_layers: int
    double_sided: bool

    def __init__(self) -> None: ...


class ModelIR:
    """Intermediate model representation."""

    name: str
    width: int
    height: int
    color_layers: int
    base_layers: int
    base_channel_idx: int
    palette: list[Channel]
    voxel_grids: list[VoxelGrid]

    def __init__(self) -> None: ...
    def num_channels(self) -> int: ...
    @staticmethod
    def Build(recipe_map: RecipeMap, db: ColorDB, cfg: BuildModelIRConfig = BuildModelIRConfig()) -> ModelIR: ...


class BuildMeshConfig:
    """Mesh build options."""

    layer_height_mm: float
    pixel_mm: float

    def __init__(self) -> None: ...


class Mesh:
    """Triangle mesh."""

    vertices: list[Vec3f]
    indices: list[Vec3i]

    def __init__(self) -> None: ...
    @staticmethod
    def Build(voxel_grid: VoxelGrid, cfg: BuildMeshConfig = BuildMeshConfig()) -> Mesh: ...


@overload
def Export3mf(path: str, model: ModelIR) -> None: ...
@overload
def Export3mf(path: str, model: ModelIR, cfg: BuildMeshConfig) -> None: ...


class CalibrationRecipeSpec:
    """Calibration recipe spec."""

    num_channels: int
    color_layers: int
    layer_order: LayerOrder

    def __init__(self) -> None: ...
    def num_recipes(self) -> int: ...
    def is_supported(self) -> bool: ...
    def recipe_at(self, index: int) -> list[int]: ...


class CalibrationFiducialSpec:
    """Fiducial layout spec."""

    offset_factor: int
    main_d_factor: int
    tag_d_factor: int
    tag_dx_factor: int
    tag_dy_factor: int

    def __init__(self) -> None: ...


class CalibrationBoardLayout:
    """Board layout settings."""

    line_width_mm: float
    resolution_scale: int
    tile_factor: int
    gap_factor: int
    margin_factor: int
    fiducial: CalibrationFiducialSpec

    def __init__(self) -> None: ...


class CalibrationBoardConfig:
    """Board configuration."""

    recipe: CalibrationRecipeSpec
    base_layers: int
    base_channel_idx: int
    layer_height_mm: float
    palette: list[Channel]
    layout: CalibrationBoardLayout

    def __init__(self) -> None: ...
    def num_recipes(self) -> int: ...
    def is_supported(self) -> bool: ...
    def has_valid_palette(self) -> bool: ...
    @staticmethod
    def for_channels(num_channels: int) -> CalibrationBoardConfig: ...


class CalibrationBoardMeta:
    """Board metadata."""

    name: str
    config: CalibrationBoardConfig
    grid_rows: int
    grid_cols: int
    patch_recipe_idx: list[int]

    def __init__(self) -> None: ...
    def num_patches(self) -> int: ...
    def save_to_json(self, path: str) -> None: ...
    @staticmethod
    def load_from_json(path: str) -> CalibrationBoardMeta: ...


def BuildCalibrationBoardMeta(cfg: CalibrationBoardConfig) -> CalibrationBoardMeta: ...
def GenCalibrationBoard(cfg: CalibrationBoardConfig, board_path: str, meta_path: str) -> None: ...
@overload
def GenColorDBFromImage(image_path: str, meta: CalibrationBoardMeta) -> ColorDB: ...
@overload
def GenColorDBFromImage(image_path: str, json_path: str) -> ColorDB: ...
