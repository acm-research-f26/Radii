## Title of Paper

An Uncertainty-guided Tiered Self-training Framework for Active Source-free Domain Adaptation in Prostate Segmentation



#### Link to Article

https://arxiv.org/pdf/2407.02893 

### Paper Summary



#### About the Model Used in this Paper

##### Dataset

This paper uses a model trained on a dataset that is not available to the current programmer. Therefore, the model had to teach itself using a different dataset that was unlabeled by picking the most useful images to learn from and train on them in stages. 


##### Architecture 

This paper uses an ML model called U-Net, which is a type of "neural network mainly used for image segmentation (dividing an image into different parts to identify a specific object)."

The architecture is a U shape which gives it its name. Typically used for medical imaging since it performs well even with smaller labeled data sets.

https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/ 

##### About UGTST

UGTST stands for Uncertainty Guided Tiered Self Training.

Model looks at unlabeled images and gives each a score based on how certain the model is about what's in the image. This is used to decide what the model trains on. 

Model also makes predictions on unlabeled data and then trains in stages or tiers on its own confident predictions. Each tier is a less confident prediction level.

##### Components

##### Important Details

### Motivation 

### Methodology

Pretrained model trains on an unfamiliar dataset where images are unlabled. Only M number of images can be labeled (this is less than total number of target images).

Then the model uses the unlabeled full target set + labeled images to adjust the pretrained source model.

Target set split into two:
- uncertain candidate set 
- assumed stable set  

Filter for diversity, removing near duplicates to keep the data varied. And then train the model on its own predictions, tier by tier. 

Calculated entropy on a single confident but wrong prediction will be low. Therefore, the uncertainty estimate is unreliable. To fix this, test-time augmentation is used. The model is run K times with random augmentations (rotation, brightness, contrast) and then averages the results. This deflates false confidence. 

Compute entropy per pixel. 

In medical segmentation, the background is the majority of the image. Therefore, all entropy values were used to build a histogram (100 bins, sorted from small to large). Find where the distribution has a peak and this becomes the confidence threshold. This threshold is self-adaptive as it is computer per image. Then average the high entropy pixels. All images are ranked by their uncertainty score. The top set of images the model is uncertain about will be in the uncertain pile, the rest are stable. 

Since MRIs are volumetric, neighboring slices look almost the same. Solution: Cluster the uncertain images by their features. Pick the most representative image from each cluster. This creates diversity and informative images. 


### Novelty

UGTST Method - Combines existing techniques like uncertainty estimation, active learning, and self training specifically for medical segmentation. The goal was to __adapt__ an existing model to a dataset that it had not seen before without the original training data. For each image, the model creates an entropy map to find what areas in image the model is most uncertain. Images that have a higher uncertainty are used to decide which images are worth training on. The model is then trained on it's own confident predictions in tiers, training on more images each stage as the confidence threshold loosens (train of images where prediction cofidence is X or above). 


### Relation to OA Paper
