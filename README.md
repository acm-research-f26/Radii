# NU-Net: Medical Ultrasound Image Segmentation

## Paper summary

Rethinking the Unpretentious U-Net for Medical Ultrasound Image Segmentation
https://arxiv.org/pdf/2209.07193 

The paper introduces NU-Net for medical image segmentation on ultrasound breast tumors. NU Net is based off of U-Net, which is a neural network designed for medical ultrasound image segmentation. Though U Net is very effective, the problem is that the model becomes very complex as people try to keep perfecting it, making it expensive and complicated to use. The authors solution was NU Net, which is a more simple architecture, built on U Net but with multiple paths. 
The authors tested NU Net on three public breast ultrasound datasets and compared it with multiple existing segmentation methods. They found that NU Net performed well on detecting breast tumors and was able to handle different sizes. They also tested it on a separate renal ultrasound dataset, where it showed that the approach could be applied to another type of ultrasound segmentation task. The authors also reported that the additional parts of NU Net increased the number of parameters without significantly increasing computational cost.

## Motivation

Medical image segmentation is identifying and outlining a specific structure or in a medical image. In this project, the goal is to learn how deep learning can be used to automatically generate accurate segmentation masks.

U-Net is useful for medical image segmentation because it can understand both the bigger picture of an image and the small details needed to identify exactly where something is. NU-Net builds on this idea while keeping the architecture relatively simple.

I selected this paper because the segmentation problem is closely related to the eventual goal of our research project, which is segmenting uterine fibroids from MRI scans.

## Novelty

The paper proposes NU-Net, a modified U-Net architecture that uses multiple feature extraction and upsampling pathways to capture information at different spatial scales. The authors aim to improve segmentation of breast tumors while avoiding the highly complex designs used by many newer U-Net variants. NU-Net uses U-Nets with different depths and shared weights, along with additional connections between features, to better handle tumors of different sizes. The goal is to improve segmentation performance while keeping the model relatively easy to understand and reproduce.

## Methodology
technical:
NU-Net is based on the U-Net architecture. The model first uses an encoder to make the image smaller and learn important features. It then uses a decoder to increase the image size again and create a segmentation mask. NU-Net adds multiple feature extraction and upsampling pathways, allowing the model to look at features at different scales. It also uses connections between different parts of the network to preserve useful image information.

data:
The authors tested NU Net on three public breast ultrasound datasets: BUSI, Dataset B, and STU. The datasets contain ultrasound images of breast lesions along with ground truth segmentation masks showing where the lesions are located. The images and masks are processed and resized before being used to train and evaluate the model. The authors also tested the model on a renal ultrasound dataset to examine how well the approach could work on another type of medical ultrasound image.

## Relation
This paper relates to our research because both projects use medical images and image segmentation to identify regions of interest. Although this paper focuses on breast ultrasound images and our project focuses on uterine fibroid MRI images, the basic goal and flow of using a deep learning model to identify and segment medical structures is similar. This paper also helps us understand how U-Net based models can be applied to medical image segmentation.




