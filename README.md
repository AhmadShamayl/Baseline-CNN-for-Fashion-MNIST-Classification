Baseline CNN Model for Fashion MNIST
This repository contains the implementation of a baseline Convolutional Neural Network (CNN) model for classifying images in the Fashion MNIST dataset. The project uses K-Fold Cross-Validation, Data Augmentation, and several performance metrics to evaluate the model.
Table of Contents
•	Introduction
•	Dataset
•	Model Architecture
•	Data Augmentation
•	Performance Metrics
•	Results
•	Usage
•	Dependencies
•	License
Introduction
This project demonstrates the use of Convolutional Neural Networks (CNNs) for image classification using the Fashion MNIST dataset. The model's performance is evaluated using K-Fold Cross-Validation and various metrics including accuracy, F1 score, precision, recall, and ROC AUC.
Dataset
The Fashion MNIST dataset is a collection of 70,000 grayscale images of 28x28 pixels each, representing 10 different classes of clothing items. The dataset is split into 60,000 training images and 10,000 testing images.
Model Architecture
The CNN model consists of the following layers:
•	Convolutional Layer: 32 filters, 3x3 kernel size, ReLU activation, He uniform initializer
•	Max Pooling Layer: 2x2 pool size
•	Flatten Layer
•	Dense Layer: 100 units, ReLU activation, He uniform initializer
•	Output Layer: 10 units, Softmax activation
The model is compiled with Stochastic Gradient Descent (SGD) optimizer, using a learning rate of 0.001 and momentum of 0.9. The loss function used is categorical cross-entropy.
Data Augmentation
Data augmentation is performed using the ImageDataGenerator class from Keras, which applies random transformations such as rotations and shifts to the training images. This helps in increasing the diversity of the training data and improving the model's generalization.
Performance Metrics
The following metrics are used to evaluate the model:
•	Accuracy: The proportion of correctly classified images.
•	F1 Score: The harmonic mean of precision and recall, providing a balance between the two.
•	Precision: The proportion of true positive predictions among all positive predictions.
•	Recall: The proportion of true positive predictions among all actual positive instances.
•	ROC AUC: The area under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish between classes.
Results
The model's performance is evaluated using 5-fold cross-validation. Learning curves are plotted for each fold, and the following average metrics are reported:
•	Accuracy: mean=91.187 std=0.260, n=5
•	F1 Score: mean=91.170 std=0.242
•	Precision: mean=91.251 std=0.194
•	Recall: mean=91.187 std=0.260
•	ROC AUC: mean=99.447 std=0.027
•	Usage
To run the project, follow these steps:
1.	Install the required dependencies:

pip install -r requirements.txt
2.	Run the test harness:

Dependencies
•	numpy
•	matplotlib
•	scikit-learn
•	keras
•	tensorflow
You can install these dependencies using pip:

pip install numpy matplotlib scikit-learn keras tensorflow
