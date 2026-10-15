"""Automated FIGO classification of uterine fibroids from MRI (UMD dataset).

Stage 1: segmentation (``figo.models``, ``figo.train``, ``figo.infer``)
Stage 2: fibroid instance separation (``figo.instances``)
Stage 3: geometric FIGO rules (``figo.stage3``, ``figo.figo_eval``)

Decision-support only: outputs are a second read for a radiologist, not a diagnosis.
"""

__version__ = "0.1.0"
