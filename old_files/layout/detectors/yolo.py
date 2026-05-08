from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from ultralytics import YOLO
import torch
import cv2
from PIL import Image

from doc2html.cfg import LayoutCfg, load_cfg
from doc2html.layout.datamodels.datamodels import Region, RegionType
from doc2html.layout.detectors.base_model import BaseModel
from doc2html.layout.utils.viz import draw_regions

log = logging.getLogger(__name__)

## Helper for parallel execution
def _run_net(net, batch_cv2, img_indices, cfg, mapping) -> Dict[int, List[Region]]: # It returns a dictionary where the index is key -> list(regions) is value
    """
    Each thread runs this: returns {global_idx: [Region, ...]} for its net.
    """
    out: Dict[int, List[Region]] = {}
    try: 
        raw = net.predict(source=batch_cv2, conf=0.2, iou=0.7, verbose=False)
        for i, res in enumerate(raw):
            gidx = img_indices[i]
            regs: List[Region] = []

            for b in res.boxes:
                label_raw = net.names[int(b.cls)]
                unified = cfg.yolo_to_unified_map.get(label_raw, label_raw)
                regs.append(
                    Region(
                        xyxy=tuple(int(v) for v in b.xyxy[0].tolist()),
                        label=RegionType(unified),
                        score=float(b.conf),
                        source=mapping[f"cuda:{net.device.index}"],
                    )
                )
            out[gidx] = regs
    except Exception as e:
        log.error("YOLO on %s failed: %s", net.device, e)
    return out

## Main Class
class YOLOModel(BaseModel):
    """
    Loads multiple YOLO weights, each on its own GPU, and returns
    `list[list[Region]]` per batch.
    """
    def load(self) -> None:
        self.mapping = {}
        self._nets: List[YOLO] = []
        devices = self.cfg.device_map.get("yolo") or list(range(torch.cuda.device_count()))
        if len(devices) < len(self.cfg.yolo_model_paths):
            raise RuntimeError("Not enough GPUs listed in device_map for YOLO models")
        for weight_path, dev_id in zip(self.cfg.yolo_model_paths, devices):
            net = YOLO(weight_path)
            net.to(f"cuda:{dev_id}")
            self.mapping[f"cuda:{dev_id}"] = Path(weight_path).stem
            self._nets.append(net)
        self._model = self._nets
        log.info("Loaded %d YOLO weights", len(self._nets))

    def unload(self) -> None:
        self._nets.clear()
        self._model = None

    @torch.no_grad()
    def predict(self, batch_cv2: List) -> List[List[Region]]:
        n_imgs = len(batch_cv2)
        if not self.is_available() or n_imgs == 0:
            return [[] for _ in batch_cv2]
        
        merged: List[List[Region]] = [[] for _ in batch_cv2]

        # Parallel over nets - each net sees the full batch
        with ThreadPoolExecutor(max_workers=len(self._nets)) as pool:
            futures = [
                pool.submit(_run_net, net, batch_cv2, list(range(n_imgs)), self.cfg, self.mapping)
                for net in self._nets
            ]
            for fut in as_completed(futures):
                piece = fut.result()
                for idx, regs in piece.items():
                    merged[idx].extend(regs)
        return merged
    
# ----------------------------------------------------------------------
# CLI entry-point
# ----------------------------------------------------------------------
import argparse

def _cli():
    ap = argparse.ArgumentParser("YOLO detector runner")
    ap.add_argument("--cfg", required=True, help="layout.yaml")
    ap.add_argument("--images", required=True, nargs="+", help="glob or list of paths")
    ap.add_argument("--save_dir", type=Path, required=True)
    ap.add_argument("--device_map", default="", help="override, e.g. 'yolo=4,5,6'")
    ap.add_argument("--device", default="cuda", help="CPU fallback for CI")
    ap.add_argument("--no_visual", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.cfg))
    if args.device_map:
        ids = [int(x) for x in args.device_map.split("=")[1].split(",")]
        cfg.device_map["yolo"] = ids

    model = YOLOModel(cfg, device=args.device)
    model.load()

    # expand globs
    img_paths: List[Path] = []
    for tok in args.images:
        img_paths += list(Path().glob(tok)) if "*" in tok else [Path(tok)]

    args.save_dir.mkdir(parents=True, exist_ok=True)

    # load cv2 batch once
    cv2_batch = [cv2.imread(str(p)) for p in img_paths]
    dets_per_img = model.predict(cv2_batch)

    for p, regs, cv2_img in zip(img_paths, dets_per_img, cv2_batch):
        json_out = args.save_dir / (p.stem + ".yolo.json")
        json_out.write_text("\n".join(r.to_json() for r in regs))

        if not args.no_visual:
            vis = draw_regions(cv2_img, regs, cfg.unified_label_viz, thickness=2)
            cv2.imwrite(str(args.save_dir / (p.stem + ".yolo.png")), vis)
        print("Saved", p.name)

    model.unload()

if __name__ == "__main__":
    _cli()