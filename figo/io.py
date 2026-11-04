"""Defensive NIfTI loading and UMD case discovery.

About 33 UMD ``*_seg.nii.gz`` files are plain (uncompressed) NIfTI despite the
``.gz`` suffix, which makes ``nibabel.load`` fail. ``load_nifti`` sniffs the
gzip magic bytes and falls back to parsing the raw bytes, so all 300 cases load.

Kaggle unpacks uploaded ``.nii.gz`` files to ``.nii``. ``resolve_nifti`` accepts
either spelling, so code that asks for ``<id>_seg.nii.gz`` also finds
``<id>_seg.nii`` (and the other way round).
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib

GZIP_MAGIC = b"\x1f\x8b"


def is_gzip(path: str | Path) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == GZIP_MAGIC


def resolve_nifti(path: str | Path) -> Path:
    """Return ``path`` if it exists, else the same file with ``.nii`` <-> ``.nii.gz`` swapped."""
    path = Path(path)
    if path.exists():
        return path
    name = path.name
    if name.endswith(".nii.gz"):
        alt = path.with_name(name[: -len(".gz")])
    elif name.endswith(".nii"):
        alt = path.with_name(name + ".gz")
    else:
        alt = None
    if alt is not None and alt.exists():
        return alt
    raise FileNotFoundError(f"{path} (also tried {alt.name if alt else 'no alternative'})")


def load_nifti(path: str | Path) -> nib.Nifti1Image:
    """Load a NIfTI file whether or not it is really gzip-compressed.

    Tries gzip first (when the magic bytes say so), otherwise parses the bytes
    as raw NIfTI. Data is fully read into memory, so the returned image does not
    depend on the file handle staying open.
    """
    path = resolve_nifti(path)
    raw = path.read_bytes()
    if raw[:2] == GZIP_MAGIC:
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError) as e:
            raise OSError(f"{path}: gzip header present but stream is corrupt ({e})") from e
    for klass in (nib.Nifti1Image, nib.Nifti2Image):
        try:
            return klass.from_bytes(raw)
        except Exception:  # wrong NIfTI version; try the next one
            continue
    raise OSError(f"{path}: not a readable NIfTI-1/NIfTI-2 file")


@dataclass(frozen=True)
class Case:
    case_id: str
    image: Path
    seg: Path | None


def _strip(name: str, suffix: str) -> str | None:
    for ext in (".nii.gz", ".nii"):
        if name.endswith(suffix + ext):
            return name[: -len(suffix + ext)]
    return None


def _find(directory: Path, stem: str) -> Path | None:
    for ext in (".nii.gz", ".nii"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def discover_cases(root: str | Path, require_seg: bool = True) -> list[Case]:
    """Find ``<id>_t2.nii[.gz]`` images (any depth) and their ``<id>_seg`` masks."""
    root = Path(root)
    cases = []
    for img in sorted(root.rglob("*_t2.nii*")):
        case_id = _strip(img.name, "_t2")
        if case_id is None:
            continue
        seg = _find(img.parent, f"{case_id}_seg")
        if seg is None and require_seg:
            raise FileNotFoundError(f"No segmentation found for {img}")
        cases.append(Case(case_id, img, seg))
    if not cases:
        raise FileNotFoundError(f"No *_t2.nii[.gz] files under {root}")
    ids = [c.case_id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case ids found; check the dataset folder for copies")
    return cases
