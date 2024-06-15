# Bodybuilding Pose Classifier

## Project Overview
The Bodybuilding Pose Classifier is a machine learning project developed as a capstone for the Machine Learning Certificate from the University of London. This model is trained to identify and classify various bodybuilding poses from images. The aim of this project is to demonstrate the application of deep learning techniques for automated image recognition in specific sports disciplines.

## Model Description
The model uses a Convolutional Neural Network (CNN) architecture to recognize specific poses from images of bodybuilders. It has been trained with training data and validated with validation data to ensure the accuracy and robustness of the model.

## Datasets
- **Training Data**: Consists of 25 images per pose category, captured from different angles and under various lighting conditions.
- **Validation Data**: Used to monitor the model's performance during training and to prevent overfitting. Contains 25 images per pose category, similar but not identical to the training images.

## Technical Details
The model was developed in Python using the following libraries and technologies:

- **TensorFlow and Keras**: For constructing, training, and validating the CNN.
- **NumPy**: For data manipulation and processing.
- **Matplotlib**: For visualizing training and validation results.
- **ImageDataGenerator from Keras**: Used for image preprocessing and data augmentation to enhance model performance by increasing the diversity of the training data.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/JumpiiX/BodybuildingPoseClassifier.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Train and Test the Model
To train and test the model, execute the following command:
```bash
python training/train_model.py
