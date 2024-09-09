# deep-learning-projects

Various neural networks for university work and online challenges

# CNN-Work

This repository includes some CNN code that I have written for a pattern recognition and analysis course that I am currently taking at university.

## Eigenfaces Analysis

This project involves comparing the effectiveness of Principal Component Analysis (PCA) combined with a Random Forest regressor against a Convolutional Neural Network (CNN) for facial recognition.

- Dataset: Labelled Faces in the Wild (LFW)
- Models:
  - PCA + Random Forest Regressor: Uses PCA to reduce dimensionality and a Random Forest to predict facial features.
  - CNN: A simple CNN with three convolutional layers, max-pooling, and dense layers.
- Objective: Match faces to names and analyze performance.

## DawnBench Challenge

In this challenge, we focus on classifying images from the CIFAR10 dataset using a CNN model that is optimized for training efficiency and accuracy.

- Dataset: CIFAR10
- Model: A CNN that achieves greater than 93% accuracy and can be trained in less than 30 minutes on UQ’s HPC.
- Classes: Airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Brain MRI Generation

This project uses Generative Adversarial Networks (GANs) to generate realistic brain MRI images.

- Dataset: OASIS
- Models:
  - Discriminative Network: Identifies fake brain MRI images.
  - Generative Network: Creates realistic brain MRI images.
- Objective: Generate high-quality brain MRI images for research purposes.
