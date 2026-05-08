from __future__ import annotations
import logging
import os
from typing import List
import argparse
from pathlib import Path
import cv2

import surya

from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from doc2html.layout.detectors.base_model import BaseModel
from doc2html.layout.datamodels.datamodels import Region, RegionType
from doc2html.cfg import LayoutCfg, load_cfg
from doc2html.layout.utils.viz import draw_regions

log = logging.getLogger(__name__)

def _run_replica(replica, slice_imgs, slice_indices, cfg):
    """
    Returns {global_index: list[Region], ...} for its slice.
    Any exception -> empty dict (logged by caller).
    """
    out: dict[int, list[Region]] = {}
    try:
        raw = replica(slice_imgs)
        for loc_i, surya_out in enumerate(raw):
            gidx = slice_indices[loc_i]
            regions: list[Region] = []
            if surya_out and getattr(surya_out, "bboxes", None):
                for box in surya_out.bboxes:
                    unified = cfg.surya_to_unified_map.get(box.label, box.label)
                    regions.append(
                        Region(
                            xyxy=tuple(int(v) for v in box.bbox),
                            label=RegionType(unified),
                            score=float(box.confidence),
                            source="surya",
                        )
                    )
            out[gidx] = regions
    except Exception as e:
        log.error("Replica on %s failed: %s", replica.device, e)
    return out

class SuryaModel(BaseModel):
    """
    Adapter around the Surya LayoutPredictor that satisfies the BaseModel
    interface and outputs `List[Region]` per input image
    """
    def __init__(self, cfg: LayoutCfg, device: str = "cuda") -> None:
        super().__init__(cfg, device)
        # Surya cares about this env variable for intra batching
        os.environ["LAYOUT_BATCH_SIZE"] = str(cfg.batch_size)

    # Loading
    def load(self) -> None:
        log.info("Loading Suyra Layout predictor on %s ...", self.device)

        gpu_ids = self.cfg.device_map.get("surya") or [0] # default to single GPU
        self._replicas = []
        for gid in gpu_ids:
            try:
                from surya.settings import settings
                settings.DISABLE_TQDM = True
                from surya.layout import LayoutPredictor as P
                pred = P(device=f"cuda:{gid}")
                self._replicas.append(pred)
            except Exception as e:
                log.error("Surya replica on cuda:%d failed: %s", gid, e)
        self._model = self._replicas if self._replicas else None

    # Unload
    def unload(self) -> None:
        # Surya has no explicit .close(): removin the object frees GPU memory
        self._predictor = None
        self._model = None

    # Inference 
    @torch.no_grad() # Surya is a torch model internally
    def predict(self, batch_pil: List[Image.Image]) -> List[List[Region]]:
        """
        Parameters
        ----------
        batch_pil: list[PIL.Image]
            Must all be RGB / uint8.
        
        Returns
        ----------
        list[list[Region]]
        Parallel to input order. Empty inner list if predictor missing / 
        errored or image invalid
        """
        n_imgs = len(batch_pil)
        if not self.is_available() or n_imgs == 0:
            return [[] for _ in batch_pil]
        
        # Round Robin Split
        idx_by_rep, imgs_by_rep = [[] for _ in self._replicas], [[] for _ in self._replicas]
        for i, img in enumerate(batch_pil):
            r = i % len(self._replicas)
            idx_by_rep[r].append(i)
            imgs_by_rep[r].append(img)

        merged: list[list[Region]] = [[] for _ in batch_pil]

        # Fire all replicas in parallel
        with ThreadPoolExecutor(max_workers=len(self._replicas)) as pool:
            futures = [
                pool.submit(_run_replica, rep, imgs_by_rep[k], idx_by_rep[k], self.cfg)
                for k, rep in enumerate(self._replicas)
                if imgs_by_rep[k]
            ]
            for fut in as_completed(futures):
                slice_dict = fut.result()
                for gidx, regs in slice_dict.items():
                    merged[gidx] = regs
        return merged
    
def _cli() -> None:
    p = argparse.ArgumentParser("Surya Detector CLI")
    p.add_argument("--cfg", required=True, help="layout.yaml")
    p.add_argument("--images", required=True, nargs="+",
                   help="glob or explicit image paths")
    p.add_argument("--save_dir", type=Path, required=True)
    p.add_argument("--device_map", default="", help="override YAML, e.g. 'surya=0,1,2'")
    p.add_argument("--device", default="cuda", help="fallback device if no map")
    p.add_argument("--no_visual", action="store_true", help="skip PNG writing")

    args = p.parse_args()
    cfg = load_cfg(Path(args.cfg))

    # Optional CLI override
    if args.device_map:
        ids = [int(x) for x in args.device_map.split("=")[1].split(",")]
        cfg.device_map["surya"] = ids
    
    model = SuryaModel(cfg, device=args.device)
    model.load()

    img_paths: list[Path] = []
    for tok in args.images:
        img_paths += list(Path().glob(tok)) if "*" in tok else [Path(tok)]
    
    args.save_dir.mkdir(parents=True, exist_ok=True)
    for img_p in img_paths:
        cv2_img = cv2.imread(str(img_p))
        if cv2_img is None:
            print("Skip Unreadable", img_p)
            continue

        det = model.predict([Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))])

        # Save json
        suffix = (img_p.stem + ".surya.json")
        json_out = args.save_dir / suffix
        json_out.write_text("\n".join(r.to_json() for r in det[0]))

        # Optional Visualization
        if not args.no_visual:
            vis = draw_regions(cv2_img, det[0], cfg.unified_label_viz, 2)
            filename = img_p.stem + ".surya.png"
            cv2.imwrite(str(args.save_dir / filename), vis)

        print("Done", img_p.name)

    model.unload()

if __name__ == "__main__":
    _cli()