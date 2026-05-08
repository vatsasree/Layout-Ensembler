"""
Surya layout detection adapter.

Wraps the `surya <https://github.com/VikParuchuri/surya>`_ LayoutPredictor
behind the :class:`~layout_ensembler.detectors.base_model.BaseModel` interface.

Supports **multi-GPU** inference by creating one predictor replica per GPU
and distributing images across them in a round-robin fashion using a
:class:`~concurrent.futures.ThreadPoolExecutor`.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import torch
from PIL import Image

from layout_ensembler.config import LayoutCfg
from layout_ensembler.datamodels.regions import Region, RegionType
from layout_ensembler.detectors.base_model import BaseModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Replica worker (module-level for picklability)
# ---------------------------------------------------------------------------
def _run_replica(
    replica,
    slice_imgs: List[Image.Image],
    slice_indices: List[int],
    cfg: LayoutCfg,
) -> Dict[int, List[Region]]:
    """
    Run a single Surya replica on its assigned slice of images.

    Returns
    -------
    dict[int, list[Region]]
        Mapping from global batch index to detected regions.
    """
    out: Dict[int, List[Region]] = {}
    try:
        raw = replica(slice_imgs)
        for loc_i, surya_out in enumerate(raw):
            gidx = slice_indices[loc_i]
            regions: List[Region] = []
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
        log.error("Surya replica on %s failed: %s", getattr(replica, "device", "?"), e)
    return out


# ---------------------------------------------------------------------------
# SuryaModel
# ---------------------------------------------------------------------------
class SuryaModel(BaseModel):
    """
    Adapter around the Surya LayoutPredictor.

    Automatically creates one predictor per GPU listed in
    ``cfg.device_map["surya"]`` and round-robin distributes each batch
    across them for throughput.

    Parameters
    ----------
    cfg : LayoutCfg
        Pipeline configuration.
    device : str
        Fallback device if ``device_map`` is empty.
    """

    def __init__(self, cfg: LayoutCfg, device: str = "cuda") -> None:
        super().__init__(cfg, device)
        os.environ["LAYOUT_BATCH_SIZE"] = str(cfg.batch_size)

    def load(self) -> None:
        log.info("Loading Surya LayoutPredictor on configured GPUs …")

        gpu_ids = self.cfg.device_map.get("surya") or [0]
        self._replicas = []

        for gid in gpu_ids:
            try:
                from surya.layout import LayoutPredictor
                from surya.settings import settings

                settings.DISABLE_TQDM = True
                pred = LayoutPredictor(device=f"cuda:{gid}")
                self._replicas.append(pred)
                log.info("  ✓ Surya replica on cuda:%d", gid)
            except Exception as e:
                log.error("  ✗ Surya replica on cuda:%d failed: %s", gid, e)

        self._model = self._replicas if self._replicas else None
        log.info("Loaded %d Surya replica(s)", len(self._replicas))

    def unload(self) -> None:
        self._replicas = []
        self._model = None

    @torch.no_grad()
    def predict(self, batch_pil: List[Image.Image]) -> List[List[Region]]:
        """
        Run layout detection on a batch of PIL images.

        Images are distributed across GPU replicas in round-robin order
        and processed in parallel threads.

        Parameters
        ----------
        batch_pil : list[PIL.Image.Image]
            Input images (must be RGB / uint8).

        Returns
        -------
        list[list[Region]]
            Parallel to input order. Empty inner list on error.
        """
        n_imgs = len(batch_pil)
        if not self.is_available() or n_imgs == 0:
            return [[] for _ in batch_pil]

        # ── Round-robin assignment ────────────────────────────────
        n_rep = len(self._replicas)
        idx_by_rep: List[List[int]] = [[] for _ in range(n_rep)]
        imgs_by_rep: List[List[Image.Image]] = [[] for _ in range(n_rep)]

        for i, img in enumerate(batch_pil):
            r = i % n_rep
            idx_by_rep[r].append(i)
            imgs_by_rep[r].append(img)

        merged: List[List[Region]] = [[] for _ in batch_pil]

        # ── Parallel execution across replicas ────────────────────
        with ThreadPoolExecutor(max_workers=n_rep) as pool:
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
