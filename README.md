# Multi-Class Retinal Disease Classification from Fundus Images

This repository contains a deep learning project to classify fundus images into 8 classes representing distinct retinal conditions. The project compares a custom CNN built from scratch against a fine-tuned, pretrained ResNet50 network.

## Disease Classes
- Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, Hypertension, Myopia, and Others.

## Directory Structure
- `custom_cnn.ipynb`: Notebook training the model from scratch.
- `resnet50_transfer.ipynb`: Notebook containing transfer learning and fine-tuning.
- `README.md`: Project summary.

## Methodology
1. **Pipeline Optimization:** Built using the `tf.data` API with lazy loading, multithreaded mapping, and prefetching to run efficiently on Kaggle's GPU P100.
2. **Class Imbalance:** Handled via custom loss calculations using balanced class weights computed from the training split.
3. **Target Encoding:** Implemented dynamic `tf.one_hot` mapping inside the preprocessing pipeline to monitor categorical classification metrics.

## Key Findings
