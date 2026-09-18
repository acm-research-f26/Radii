### Paper Summary
In medical deep learning predictions have to have reliable uncertainty estimates so that doctors know when to trust an automated result. While Bayesian approximates like Monte Carlo dropout method are used to estimate predictive uncertainty, this paper shows that deep regression networks systematically underestimate their variance(σ^2) which ends up giving overconfident and miscalibrated predictions.
To resolve this, the authors recommend σ

#### Title of Paper

Well-Calibrated Regression Uncertainty in Medican Imaging with Deep Learning(MIDL 2020/ MPLR 121:393-412)

Authors: Max-Heinrich Laves, Sontje Ihler, Jacob F. Fast, L¨uder A. Kahrs, Tobias Ortmaier

#### Link to Article

https://proceedings.mlr.press/v121/laves20a/laves20a.pdf


### Motivation 
In medical regression tasks like age estimation, organ boundary segmentation, and instrument pose estimation tend to have reliable predictive uncertainty. This occurs because deep Bayesian neural networks tend to be overconfident. Because there models overfit the training data, the predictive variance is usually because of an overfitted mean that causes predictive uncertainty to be underestimated.



### Novelty

### Methodology

### Relation to OA Paper
