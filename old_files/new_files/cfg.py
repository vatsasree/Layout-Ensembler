from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field, model_validator

# 1. Leaf / helper type aliases
RGB = Tuple[int, int, int]

class EnsembleCfg(BaseModel):
    """
    Validation schema for the ensemble config -> YAML
    """
    base_model_name: str 
    model_priority_order: List[str] = Field(min_length=1)
    iou_threshold: float = Field(0.0, le=1.0)

    # Optional
    protected_classes_from_overlap_removal: List[str] = Field(default_factory=list)
    conf_thresholds_for_supplementary_models: Dict[str, float] = Field(default_factory=dict)
    filtered_classes_to_keep: Dict[str, List[str]] = Field(default_factory=dict)

    # ----------- Sanity check ------------
    @model_validator(mode="after")
    def check_consistency(self) -> "EnsembleCfg":
        return self

# 2. Main Config Object
class LayoutCfg(BaseModel):
    # ---- paths -----------------
    input_base_path: Path
    ocr_results_sub_path: Path
    layout_results_sub_path: Path
    text_attributes_sub_path: Path

    # --- model names ------- #
    table_model_name: Path
    generic_model_name: Path

    # --- prompts & label maps -------- #
    multi_prompt_map: Dict[str, Path]
    yolo_to_unified_map: Dict[str, str]
    surya_to_unified_map: Dict[str, str]

    # --- labels to be directly added #
    direct_add_labels: List[str]

    # --- visualization maps ------- #
    unified_label_viz: Dict[str, RGB]

    # --- detector weights ------ #
    yolo_model_paths: List[Path]

    # --- device mapping ------ #
    device_map: Dict[str, List[int]] = Field(default_factory=dict)

    # --- models mapping ------- #
    yolo: Dict[str, bool]
    surya: Dict[str, bool]

    # --- numeric knobs ------ #
    box_padding:                 int = Field(gt=0)
    nms_iou_threshold:           float = Field(0.0, le=1.0)
    text_enclosure_threshold:    float                   = Field(0.0, le=1.0)
    indicdlp_add_iou_threshold:  float                   = Field(0.0, le=1.0)
    batch_size:                  int                     = Field(gt=0)

    # --- ensemble config ------ #
    ensemble_config: EnsembleCfg

    # Custom validation for model paths
    @model_validator(mode="after")
    def check_model_paths_exist(self) -> "LayoutCfg":
        missing = [p for p in self.yolo_model_paths if not p.exists()]
        if missing: 
            raise ValueError(f"Model Weights not found: {missing}")
        return self

class MergeCfg(BaseModel):
    html_blocks_base_dir: Path
    output_base_dir: Path
    template_path: Path
    source_subhtmls: str

    @model_validator(mode="after")
    def check_model_paths_exist(self) -> "MergeCfg":
        # Check if all the specified paths exist
        missing = [str(p) for p in [self.html_blocks_base_dir, self.output_base_dir, self.template_path] if not p.exists()]
        if missing:
            raise ValueError(f"The following paths do not exist: {missing}")
        return self
    
class FontCfg(BaseModel):
    layout_json_base_dir: Path
    output_path_dir: Path
    js_script_path: Path
    js_script_path: Path

    margin_percentage: float
    overflow_reduction_percentage: float
    categories_to_skip_font_sizing: List[int]

    min_font_size: int
    max_font_size: int
    cell_padding: int
    line_height: float
    avg_char_width_ratio: float
    filename_prefix_filter: List[str]

    @model_validator(mode='after')
    def check_model_paths_exists(self) -> "FontCfg":
        missing = [str(p) for p in [self.layout_json_base_dir, self.js_script_path] if not p.exists()]
        if missing: 
            raise ValueError(f"The following paths do not exist: {missing}")
        return self


# --------------
# 4. Convenience Loader 
# --------------
def load_cfg(path: Path) -> LayoutCfg:
    """
    Load a YAML file into a strongly-typed LayoutCfg
    """
    import yaml
    data = yaml.safe_load(path.read_text())
    return LayoutCfg.model_validate(data)

def merge_cfg(path: Path) -> MergeCfg:
    """
    Loads merge YAML
    """
    import yaml
    data = yaml.safe_load(path.read_text())
    return MergeCfg.model_validate(data)

def font_cfg(path: Path) -> FontCfg:
    import yaml
    data = yaml.safe_load(path.read_text())
    return FontCfg.model_validate(data)