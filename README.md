# ISM-winter-2020 - Skin Lesion Analysis Towards Melanoma Detection Tumor Detection
This repo contains the solution used to solve the Intelligent Systems in Medicine course`s practical challenge. The solution was done using python, tensorflow, sklearn, matplotlib and opencv.

## Description: 
Melanoma is the deadliest form of skin cancer but curable if detected in an early stage. Dermoscopic images can help improve the diagnosis of skin lesions. The ISIC challenge SSkin Lesion Analysis Towards Melanoma Detection”deals with automated classification of skin lesion images to improve algorithms and cancer detection.

## The task: 
The task is the classification of skin lesion images. A file with the split between training and validation data will be uploaded, as well as a link to the training images. DO NOT use validation data for training. The algorithms and networks should be able to predict the class based on an input image. Feel free to use the ISIC data provided for segmentation tasks from previous years. The data comes with a few challenges that you need to figure out.

## What to solve:
• Perform skin lesion classification with classical feature extraction
• Perform skin lesion classification with neural networks

## Approach

### Classical Approach
Preprocessing steps:
* Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image quality.
* Hair Removal to remove unwanted elements from the images.
* Segmentation using k-means to isolate the region of interest.
* Circular mask to focus on the central part of the skin lesion.

Feature extraction:
* Border analysis to detect irregular borders by comparing with regular shapes (e.g., rectangle).
* Diameter estimation using contour features to measure the lesion's size.
* Color Histogram to capture color distribution information.
* Moments to describe the spatial distribution of pixel intensities.
* Hu-Moments to represent shape characteristics of the lesion.
* Co-Occurrence matrix to analyze texture patterns in the image.

Machine learning models and accuracies:
* Support Vector Machine (SVM) achieved 62% accuracy in classification.
* k-Nearest Neighbors (KNN) achieved 60% accuracy in classification.
* Linear Regression achieved 58% accuracy in classification.
* Random Forest achieved 48% accuracy in classification.

Observations: SVM outperformed other models, providing the highest accuracy for skin lesion classification. The chosen features and models could be further optimized for improved performance. Further research and experimentation may enhance the overall classification accuracy.

### Modern- Deep Learning Approach

* Transfer Learning with Ensembling: The solution to the challenge involved using a transfer learning approach with ensembling techniques to improve classification accuracy. Transfer learning enables leveraging pre-trained models to tackle new tasks with less data, while ensembling combines multiple models' predictions for better performance.
* Models Used: Two powerful pre-trained models were employed for this task: Inception Net V3 and DenseNet. These models are well-known for their capabilities in image recognition tasks and are commonly used as feature extractors.
* Overall Accuracy: The ensemble of Inception Net V3 and DenseNet resulted in an impressive 60% overall accuracy for the challenge. This indicates the effectiveness of combining the strengths of these models to enhance the classification results.
* Continuous Improvement: While achieving a 60% accuracy is a significant accomplishment, there is always room for improvement. Fine-tuning the model's hyperparameters and exploring different ensemble strategies might further enhance the accuracy.
* Generalization: The models trained with transfer learning can generalize well to new data and may prove valuable for related image classification tasks beyond the current challenge.

Conclusion: To push the performance even further, experimenting with other state-of-the-art models, data augmentation techniques, or exploring different ensembling methods could be promising avenues for future research. Continual refinement of the model can lead to more accurate and robust results in image classification tasks.

## Disclaimer

The contents of this git account is the property of TUHH. The files can be used to view and analyze my skillsets. Any other purpose is considered illegal.
