#!/usr/bin/env python3
"""
Real-time automated analysis and reporting of uterine MRI.

Pipeline (per the methodology of paper):
  1. (optional) stream slices from scanner (ISMRMRD) -> stack -> NIfTI
  2. nnU-Net v2 3D full-res ResEnc-M, 5-fold ensemble (softmax averaging)
  3. Volumetry per class (voxel count x voxel volume -> cm^3)
  4. 3D connected components for myomas and Nabothian cysts (SciPy)
  5. Structured HTML report with max-volume representative slices
  6. Evaluation: Dice / IoU per structure

CLI (Fire):
  python uterine_mri_tool.py prepare_dataset  --images_dir ... --labels_dir ...
  python uterine_mri_tool.py run              --input scan.nii.gz --output_dir out/
  python uterine_mri_tool.py run_from_ismrmrd --h5 raw.h5 --output_dir out/
  python uterine_mri_tool.py evaluate         --pred_dir ... --gt_dir ...

Requirements:
  pip install nnunetv2 nibabel scipy numpy matplotlib fire torch
  pip install ismrmrd   # only for streaming/raw-data conversion
"""

# imports

import base64
import io
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import fire
import nibabel as nib
import numpy as np
from scipy import ndimage

# Config

DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID}_UterusMRI"
LABELS = {"background": 0, "uterine_wall": 1, "uterine_cavity": 2,
          "uterine_myoma": 3, "nabothian_cyst": 4}
CLASS_NAMES = {v: k for k, v in LABELS.items() if v > 0}
LESION_CLASSES = (3, 4)  # myoma, Nabothian cyst
# ResEnc-M with default plans identifier created by nnUNetPlannerResEncM
TRAINER = "nnUNetTrainer"
PLANS = "nnUNetResEncUNetMPlans"
CONFIG = "3d_fullres"
FOLDS = (0, 1, 2, 3, 4)  # folds to ensemble at inference
OVERLAY_COLORS = {1: (0.2, 0.6, 1.0), 2: (1.0, 0.9, 0.1),
                  3: (1.0, 0.2, 0.2), 4: (0.2, 1.0, 0.4)}


def _nnunet_dirs():
    raw = os.environ.get("nnUNet_raw")
    res = os.environ.get("nnUNet_results")
    if not raw or not res:
        raise EnvironmentError("Set nnUNet_raw, nnUNet_preprocessed and nnUNet_results.")
    return Path(raw), Path(res)


# 1. Dataset prep (nnU-Net v2 raw format)

def prepare_dataset(images_dir, labels_dir, test_ids=None, image_suffix=".nii.gz"):
    """
    Convert paired T2 sagittal volumes + label maps to nnU-Net raw layout.
    Files in images_dir and labels_dir must share the same stem (case id).
    test_ids: optional list/file of case ids held out for testing (e.g. 60 cases);
              these go to imagesTs/labelsTs, the rest (240) to training.
              nnU-Net's 5-fold CV then handles the train/validation split.
    """
    raw, _ = _nnunet_dirs()
    out = raw / DATASET_NAME
    for d in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (out / d).mkdir(parents=True, exist_ok=True)

    if isinstance(test_ids, (str, Path)):
        test_ids = Path(test_ids).read_text().split()
    test_ids = set(test_ids or [])

    n_train = 0
    for img in sorted(Path(images_dir).glob(f"*{image_suffix}")):
        cid = img.name[: -len(image_suffix)]
        lab = Path(labels_dir) / img.name
        if not lab.exists():
            print(f"[warn] no label for {cid}, skipping")
            continue
        split = "Ts" if cid in test_ids else "Tr"
        shutil.copy(img, out / f"images{split}" / f"{cid}_0000.nii.gz")
        shutil.copy(lab, out / f"labels{split}" / f"{cid}.nii.gz")
        n_train += split == "Tr"

    dataset_json = {
        "channel_names": {"0": "T2"},
        "labels": LABELS,
        "numTraining": n_train,
        "file_ending": ".nii.gz",
    }
    (out / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"Prepared {out} with {n_train} training cases.")
    print("Next: bash train_nnunet.sh")


# 2. Segmentation (5-fold ensemble inference)

def segment(input_nifti, output_nifti, model_dir=None, device="cuda"):
    """Run nnU-Net v2 ResEnc-M 3D full-res; softmax averaged across 5 folds."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    if model_dir is None:
        _, res = _nnunet_dirs()
        model_dir = res / DATASET_NAME / f"{TRAINER}__{PLANS}__{CONFIG}"

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=(device == "cuda"),
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=FOLDS, checkpoint_name="checkpoint_final.pth"
    )

    # nnU-Net expects files named <case>_0000.nii.gz; stage a temp copy

    tmp = Path(output_nifti).parent / "_tmp_in"
    tmp.mkdir(parents=True, exist_ok=True)
    staged = tmp / "case_0000.nii.gz"
    shutil.copy(input_nifti, staged)
    out_dir = Path(output_nifti).parent / "_tmp_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor.predict_from_files(
        [[str(staged)]], [str(out_dir / "case.nii.gz")],
        save_probabilities=False, overwrite=True,
        num_processes_preprocessing=1, num_processes_segmentation_export=1,
    )
    shutil.move(str(out_dir / "case.nii.gz"), output_nifti)
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    return output_nifti

# 3. Volumetry + lesion enumeration

def compute_volumes(mask, voxel_dims_mm):
    """Return {class_name: volume_cm3}. Volume = n_voxels * prod(spacing) / 1000."""
    vox_mm3 = float(np.prod(voxel_dims_mm))
    return {name: float((mask == lab).sum()) * vox_mm3 / 1000.0
            for lab, name in CLASS_NAMES.items()}


def analyze_lesions(mask, voxel_dims_mm):
    """3D connected components (SciPy) per lesion class; per-lesion stats."""
    vox_mm3 = float(np.prod(voxel_dims_mm))
    out = {}
    for lab in LESION_CLASSES:
        labeled, n = ndimage.label(mask == lab)  # default 6-connectivity
        lesions = []
        for i in range(1, n + 1):
            comp = labeled == i
            nvox = int(comp.sum())
            com = ndimage.center_of_mass(comp)
            lesions.append({
                "id": i,
                "volume_cm3": nvox * vox_mm3 / 1000.0,
                "n_voxels": nvox,
                "centroid_vox": [round(float(c), 1) for c in com],
            })
        lesions.sort(key=lambda d: -d["volume_cm3"])
        out[CLASS_NAMES[lab]] = {"count": n, "lesions": lesions}
    return out

# 4. HTML report

def _slice_axis(spacing):
    """Sagittal stacks: slice (through-plane) axis is usually the coarsest spacing."""
    return int(np.argmax(spacing))


def _render_slice(img2d, mask2d, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = np.percentile(img2d, (1, 99))
    norm = np.clip((img2d - lo) / (hi - lo + 1e-8), 0, 1)
    rgb = np.stack([norm] * 3, -1)
    overlay = rgb.copy()
    for lab, col in OVERLAY_COLORS.items():
        overlay[mask2d == lab] = col
    blended = 0.65 * rgb + 0.35 * overlay

    fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
    ax[0].imshow(np.rot90(rgb)); ax[0].set_title("T2 sagittal"); ax[0].axis("off")
    ax[1].imshow(np.rot90(blended)); ax[1].set_title(title); ax[1].axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def generate_report(image_nifti, mask_nifti, report_path, timing=None, patient_id="N/A"):
    img_nii, msk_nii = nib.load(image_nifti), nib.load(mask_nifti)
    img = np.asarray(img_nii.get_fdata(), dtype=np.float32)
    msk = np.asarray(msk_nii.dataobj).astype(np.uint8)
    spacing = tuple(float(s) for s in img_nii.header.get_zooms()[:3])

    vols = compute_volumes(msk, spacing)
    lesions = analyze_lesions(msk, spacing)
    ax = _slice_axis(spacing)

    # representative slice = slice with max area for each structure

    figs = []
    for lab, name in CLASS_NAMES.items():
        other = tuple(i for i in range(3) if i != ax)
        areas = (msk == lab).sum(axis=other)
        if areas.max() == 0:
            continue
        idx = int(np.argmax(areas))
        s_img = np.take(img, idx, axis=ax)
        s_msk = np.take(msk, idx, axis=ax)
        figs.append((name, idx, _render_slice(
            s_img, s_msk, f"{name.replace('_', ' ')} (slice {idx})")))

    legend = "".join(
        f'<span style="display:inline-block;width:12px;height:12px;background:'
        f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)});margin:0 4px 0 12px"></span>'
        f'{CLASS_NAMES[l].replace("_", " ")}' for l, c in OVERLAY_COLORS.items())

    vol_rows = "".join(f"<tr><td>{k.replace('_',' ')}</td><td>{v:.2f}</td></tr>"
                       for k, v in vols.items())
    lesion_sections = ""
    for name, d in lesions.items():
        rows = "".join(f"<tr><td>{l['id']}</td><td>{l['volume_cm3']:.2f}</td></tr>"
                       for l in d["lesions"]) or '<tr><td colspan="2">None detected</td></tr>'
        lesion_sections += (f"<h3>{name.replace('_',' ').title()} &mdash; count: {d['count']}</h3>"
                            f"<table><tr><th>#</th><th>Volume (cm&sup3;)</th></tr>{rows}</table>")
    fig_html = "".join(
        f'<div class="fig"><h4>{n.replace("_"," ").title()} &mdash; max-volume slice {i}</h4>'
        f'<img src="data:image/png;base64,{b}"/></div>' for n, i, b in figs)
    timing_html = (f"<p>Processing time: {timing:.1f} s</p>" if timing else "")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Uterine MRI Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:900px;margin:24px auto;color:#222}}
table{{border-collapse:collapse;margin:8px 0}}th,td{{border:1px solid #ccc;padding:6px 14px;text-align:left}}
th{{background:#f0f0f0}}.fig{{margin:16px 0}}.note{{font-size:12px;color:#a00;margin-top:24px}}
</style></head><body>
<h1>Automated Uterine MRI Report</h1>
<p>Patient/Case: {patient_id} &nbsp;|&nbsp; Generated: {datetime.now():%Y-%m-%d %H:%M:%S}
&nbsp;|&nbsp; Voxel size (mm): {spacing[0]:.2f} &times; {spacing[1]:.2f} &times; {spacing[2]:.2f}</p>
{timing_html}
<h2>Volumetry</h2><table><tr><th>Structure</th><th>Volume (cm&sup3;)</th></tr>{vol_rows}</table>
<h2>Incidental Findings</h2>{lesion_sections}
<h2>Segmentation Visualization</h2><div>{legend}</div>{fig_html}
<p class="note">Automatically generated; requires review by a qualified radiologist.
Not for stand-alone diagnostic use.</p>
</body></html>"""
    Path(report_path).write_text(html)
    return {"volumes_cm3": vols, "lesions": lesions}


# 5. Raw-data streaming -> NIfTI (simplified; adapt to your sequence/protocol)

def slices_to_nifti(slices, spacing_mm, out_path):
    """
    Stack 2D slices (list of HxW arrays, in acquisition order) into a NIfTI.
    spacing_mm: (in-plane row, in-plane col, slice thickness/gap).
    """
    vol = np.stack(slices, axis=-1).astype(np.float32)
    affine = np.diag([spacing_mm[0], spacing_mm[1], spacing_mm[2], 1.0])
    nib.save(nib.Nifti1Image(vol, affine), out_path)
    return out_path


def ismrmrd_to_nifti(h5_path, out_path, group="dataset"):
    """
    Minimal Cartesian reconstruction from an ISMRMRD file: per-slice inverse FFT
    and root-sum-of-squares coil combination. Intended as a starting point:
    in a live deployment, acquisitions arrive slice-by-slice via the streaming
    server and you reconstruct each as it completes. Parallel imaging, partial
    Fourier, oversampling removal and orientation handling must be added to
    match your protocol.
    """
    import ismrmrd
    from collections import defaultdict

    ds = ismrmrd.Dataset(h5_path, group, create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(ds.read_xml_header())
    enc = header.encoding[0]
    fov, mat = enc.reconSpace.fieldOfView_mm, enc.reconSpace.matrixSize
    spacing = (fov.x / mat.x, fov.y / mat.y, fov.z / max(mat.z, 1))

    lines = defaultdict(dict)  # slice -> {ky: [coils, nx]}
    for i in range(ds.number_of_acquisitions()):
        acq = ds.read_acquisition(i)
        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            continue
        lines[acq.idx.slice][acq.idx.kspace_encode_step_1] = acq.data

    slices = []
    for s in sorted(lines):
        ks = lines[s]
        ky_max = max(ks) + 1
        coils, nx = next(iter(ks.values())).shape
        kspace = np.zeros((coils, ky_max, nx), dtype=np.complex64)
        for ky, d in ks.items():
            kspace[:, ky, :] = d
        img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(1, 2)),
                                           axes=(1, 2)), axes=(1, 2))
        slices.append(np.sqrt((np.abs(img) ** 2).sum(0)))
    return slices_to_nifti(slices, spacing, out_path)


# 6. End-to-end entry points

def run(input, output_dir, model_dir=None, device="cuda", patient_id="N/A"):
    """Full pipeline on a NIfTI: segment -> volumetry/lesions -> HTML report."""
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask_path = str(out / "segmentation.nii.gz")
    segment(input, mask_path, model_dir=model_dir, device=device)
    elapsed = time.time() - t0
    res = generate_report(input, mask_path, out / "report.html",
                          timing=elapsed, patient_id=patient_id)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"Done in {elapsed:.1f}s -> {out/'report.html'}")
    return res


def run_from_ismrmrd(h5, output_dir, **kw):
    """Raw ISMRMRD -> NIfTI -> full pipeline."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    nii = ismrmrd_to_nifti(h5, str(out / "t2_sag.nii.gz"))
    return run(nii, output_dir, **kw)

# 7. Evaluation: Dice and IoU per structure

def dice_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    p, g = pred.sum(), gt.sum()
    if p + g == 0:  # structure absent in both (e.g. healthy case): skip
        return None, None
    dice = 2 * inter / (p + g)
    iou = inter / (p + g - inter)
    return float(dice), float(iou)


def evaluate(pred_dir, gt_dir, out_json=None):
    """Per-case, per-structure Dice/IoU; averaged across cases per label."""
    scores = {name: {"dice": [], "iou": []} for name in CLASS_NAMES.values()}
    for pf in sorted(Path(pred_dir).glob("*.nii.gz")):
        gf = Path(gt_dir) / pf.name
        if not gf.exists():
            continue
        pred = np.asarray(nib.load(pf).dataobj).astype(np.uint8)
        gt = np.asarray(nib.load(gf).dataobj).astype(np.uint8)
        for lab, name in CLASS_NAMES.items():
            d, i = dice_iou(pred == lab, gt == lab)
            if d is not None:
                scores[name]["dice"].append(d)
                scores[name]["iou"].append(i)
    summary = {n: {"dice_mean": float(np.mean(v["dice"])) if v["dice"] else None,
                   "iou_mean": float(np.mean(v["iou"])) if v["iou"] else None,
                   "n_cases": len(v["dice"])} for n, v in scores.items()}
    print(json.dumps(summary, indent=2))
    if out_json:
        Path(out_json).write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    fire.Fire({
        "prepare_dataset": prepare_dataset,
        "run": run,
        "run_from_ismrmrd": run_from_ismrmrd,
        "evaluate": evaluate,
    })