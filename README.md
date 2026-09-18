# RADII

## The Problem

Uterine fibroids are noncancerous muscular growths in the wall of the uterus, and they are extremely common: they affect up to 70 to 80% of women by age 50, and cause heavy bleeding, pelvic pain, and fertility problems. A fibroid's size is one of the most consequential numbers in its care. It helps decide whether a patient qualifies for less invasive treatments, it is used to enroll patients in clinical trials, and, tracked over time, it is how clinicians judge whether a treatment is actually shrinking the fibroid.

In everyday practice that size is estimated quickly, by measuring a few diameters and applying a simple formula, rather than by tracing the fibroid in full. The problem is that this shortcut can be significantly inaccurate for fibroids that are irregular in shape. A 2025 study found that the quick estimate differed from a full, careful measurement by more than 20% in about 1 in 5 fibroids, and by more than 30% in about 1 in 10, with the largest difference reaching roughly 56%, even though the average error looked almost perfect. An error that large matters, because treatment success is often judged by whether a fibroid shrinks 20 to 30%, so a measurement that is off by that much can turn a real result into a false one, or hide a real one. Today a clinician has no reliable way to know, at the moment of measurement, whether the estimate they just recorded can be trusted.

## What RADII Is

RADII (a nod to the radii and diameters used to size a fibroid) studies the reliability of this everyday fibroid measurement. Using MRI scans that have already been outlined by expert radiologists, we treat the full traced outline as the true size and compare it against the quick clinical estimate for each fibroid. We then look at which measurable shape characteristics, such as elongation, lobulation, and overall irregularity, make the shortcut fail, and we build lightweight models that flag the fibroids whose measurement cannot be trusted. The goal is not to replace a radiologist's judgment, but to support it: to give an honest sense of how reliable an existing measurement is, so a quick estimate can be trusted when it is safe and double-checked when it is not.

## The Dataset

We use the public Uterine Myoma MRI Dataset (UMD), the largest openly available uterine fibroid MRI dataset. It contains sagittal T2-weighted pelvic MRI scans from 300 patients, spanning all fibroid types, each with expert pixel-level segmentation masks that label the uterine wall, cavity, fibroids, and cysts. Because these masks give an exact, traced measurement of every fibroid for free, and let us compute the quick clinical estimate from the same scan, UMD lets us measure the error of the shortcut directly, at scale, without needing to collect or annotate any new data ourselves.

## Related Models

* **VM-UNet:** Implementation and documentation located in [`models/vmunet/`](./models/vmunet/README.md).
