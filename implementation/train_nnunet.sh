#!/usr/bin/env bash
# nnU-Net v2, 3D full-res, ResEnc-M, 5-fold CV, default settings (1000 epochs), from scratch.

set -euo pipefail

export nnUNet_raw=${nnUNet_raw:-$HOME/nnUNet_raw}
export nnUNet_preprocessed=${nnUNet_preprocessed:-$HOME/nnUNet_preprocessed}
export nnUNet_results=${nnUNet_results:-$HOME/nnUNet_results}
DS=501

# 1. Fingerprint + plan with the ResEnc-M planner, then preprocess
nnUNetv2_plan_and_preprocess -d $DS -pl nnUNetPlannerResEncM --verify_dataset_integrity

# 2. Train five folds (run in parallel across GPUs if available)
for FOLD in 0 1 2 3 4; do
  nnUNetv2_train $DS 3d_fullres $FOLD -p nnUNetResEncUNetMPlans
done

# step 3 and 4 were skipped as this preprocessing was already conducted

# 3. Optional: find best config / ensemble checking
# nnUNetv2_find_best_configuration $DS -c 3d_fullres -p nnUNetResEncUNetMPlans

# 4. Batch inference on the held-out test set (5-fold softmax averaging is the default)
# nnUNetv2_predict -i $nnUNet_raw/Dataset501_UterusMRI/imagesTs -o preds_test \
#   -d $DS -c 3d_fullres -p nnUNetResEncUNetMPlans -f 0 1 2 3 4