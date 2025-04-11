# TensorFlow Practice

A series of experiments focused on building and evaluating deep learning models using TensorFlow.

## Overview

The goal is to build, train, evaluate, optimize, and compare deep learning models in TensorFlow for optimal classification performance. This involves preprocessing datasets, experimenting with normalization, data augmentation, and encoding techniques, as well as implementing various neural network architectures, including fully connected and convolutional networks. The two primary datasets used are **EMNIST** for handwritten letter classification and **CIFAR-10** for object classification.

## Objective

The objective is to gain hands-on experience with different neural network architectures, hyperparameters, and techniques, specifically tailored to image classification tasks. By working with these datasets, the goal is to better understand how various neural network models perform under different configurations and optimizations. The implementation follows an object-oriented programming (OOP) approach, ensuring modularity and reusability across various experiments.

## Tasks

#### Handwritten Letters Classification: EMNIST

Eight experiments were conducted, exploring a variety of neural network architectures, including different numbers of hidden layers and neurons, as well as several activation functions such as ReLU, GELU, Sigmoid, and Tanh. The evaluation of models was carried out using metrics like accuracy, precision, recall, and loss. The key findings from this task revealed that a neural network with a single hidden layer of 512 neurons performed most effectively, delivering the best results. Early epochs showed significant improvement in accuracy, suggesting that the model quickly learned useful features from the dataset. However, challenges arose when attempting to classify certain letter pairs, such as 'i' and 'l' or 'q' and 'g', as their handwritten forms were highly similar and often difficult to differentiate.

#### CNN for Image Classification: CIFAR-10

Four experiments were conducted, exploring model architectures that included Convolutional Neural Networks (CNNs) with 2-3 convolution layers, various activation functions, batch normalization, and dropout. Performance was evaluated using the same metrics: accuracy, precision, recall, and loss. Key insights from this task included the effectiveness of the ELU activation function and the selective use of dropout, both of which significantly enhanced model performance. The final model achieved an accuracy of 81%, demonstrating solid performance. However, overfitting remained a challenge, necessitating careful tuning of dropout rates and the introduction of batch normalization to stabilize training. The most successful configuration involved a CNN with three convolution layers, ELU activation, selective dropout, and batch normalization.

## Important

I love PyTorch... Yes PyTorch.
