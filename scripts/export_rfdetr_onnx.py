#!/usr/bin/env python3
"""Export a .recomodel (RF-DETR checkpoint bundle) to ONNX for reco-detect.

reco's `CpuDetrDetector` consumes a plain RF-DETR ONNX export: two named
outputs, `dets` (boxes, cxcywh, normalized to the export resolution) and
`labels` (raw per-class logits, sigmoid applied downstream). No letterbox,
no NMS - both are reco-detect's job (see detectors/mod.rs::postprocess_detr).

Run with a venv that has `rfdetr` (and its `onnx`/`torch` deps) installed,
e.g. an existing reco-training venv:

    /path/to/reco-training/.runtime/venv/bin/python3 \\
        scripts/export_rfdetr_onnx.py \\
        --recomodel /path/to/Basketball-Small-*.recomodel \\
        --out /tmp/basketball-rfdetr.onnx

Model files are gitignored in this repo (see .gitignore: *.onnx, *.pt,
*.engine) - this script's output is for local testing, not committed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recomodel", required=True, type=Path, help="Path to the .recomodel bundle")
    parser.add_argument("--out", required=True, type=Path, help="Output .onnx path")
    parser.add_argument("--resolution", type=int, default=512, help="Export resolution (must match training)")
    args = parser.parse_args()

    if not args.recomodel.exists():
        raise SystemExit(f"recomodel not found: {args.recomodel}")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        with zipfile.ZipFile(args.recomodel) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            zf.extract("weights/checkpoint_best_total.pth", tmp)

        checkpoint_path = tmp / "weights" / "checkpoint_best_total.pth"
        class_names = manifest["classes"]
        architecture = manifest["runtime"]["architecture"]
        if architecture != "rf-detr":
            raise SystemExit(f"Unsupported architecture {architecture!r}; this script only handles rf-detr")

        print(f"Exporting {manifest['displayName']} (classes={class_names}) from {checkpoint_path.name}")

        from rfdetr import RFDETRSmall

        model = RFDETRSmall(
            pretrain_weights=str(checkpoint_path),
            num_classes=len(class_names),
            resolution=args.resolution,
            trust_checkpoint=True,
        )

        export_dir = tmp / "export"
        output_path = Path(model.export(output_dir=str(export_dir), format="onnx"))
        print(f"Exported: {output_path}")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(output_path, args.out)
        print(f"Copied to: {args.out}")
        print(
            f"class_names={class_names} resolution={args.resolution} "
            f"background_class_index={len(class_names)} (last exported slot)"
        )


if __name__ == "__main__":
    main()
