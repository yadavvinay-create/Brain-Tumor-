# Brain Tumor Classification using SVM

This project demonstrates a brain tumor classification system using Support Vector Machines (SVM) on MRI images. The system preprocesses images, applies dimensionality reduction using Principal Component Analysis (PCA), and then trains an SVM model to classify brain tumors into four categories: glioma, meningioma, notumor, and pituitary.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Brain tumor diagnosis is a critical task in medical imaging. This project aims to automate the classification of brain tumors from MRI scans using machine learning techniques. We leverage image processing, feature extraction with PCA, and a robust SVM classifier to achieve accurate predictions.

## Features

- **Image Loading and Preprocessing**: Loads images from specified directories and resizes them to a uniform size (64x64 pixels). Images are normalized to a 0-1 range.
- **Data Splitting**: Splits the dataset into training and testing sets.
- **Feature Scaling**: Applies `StandardScaler` to normalize the flattened image data.
- **Dimensionality Reduction**: Utilizes PCA to reduce the high-dimensional image data into a smaller, more manageable set of principal components. This helps in reducing computational complexity and potentially improving model performance.
- **SVM Classification**: Trains a Support Vector Machine (SVM) classifier with a linear kernel on the PCA-transformed data.
- **Model Evaluation**: Provides a classification report including precision, recall, f1-score, and overall accuracy.
- **Single Image Prediction**: Includes a utility to preprocess and predict the class of a single new image.

## Dataset

The dataset is expected to be organized into a `Training` directory, with subdirectories for each tumor type. The categories are:
- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

The notebook assumes the data is located in `drive/MyDrive/brain/Training`. Please adjust the `base_dir` variable in the notebook if your dataset path is different.

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install the required libraries:**
    You can install the necessary Python packages using pip:
    ```bash
    pip install numpy matplotlib opencv-python scikit-learn tensorflow
    ```
    Note: If you are using Google Colab, most of these libraries are pre-installed.

## Usage

1.  **Upload the Notebook to Google Colab (Recommended):**
    This notebook is designed to be run in Google Colab, especially if your dataset is in Google Drive.
    * Open Google Colab.
    * Go to `File` -> `Upload notebook` and select `Brain Tumor project.ipynb`.
    * Mount your Google Drive to access the dataset:
        ```python
        from google.colab import drive
        drive.mount('/content/drive')
        ```
2.  **Place your dataset:**
    Ensure your `Training` dataset (containing `glioma`, `meningioma`, `notumor`, `pituitary` subdirectories) is located at `drive/MyDrive/brain/Training` or update the `base_dir` variable in the notebook accordingly.

3.  **Run all cells:**
    Execute each cell in the Jupyter Notebook sequentially.
    * `Cell 1-3`: Import libraries and set up data paths.
    * `Cell 4-7`: Load images, preprocess them, and split the data.
    * `Cell 8-10`: Flatten images, apply StandardScaler, and perform PCA.
    * `Cell 11`: Train the SVM model.
    * `Cell 12-14`: Make predictions and print the classification report and accuracy score.
    * `Cell 15`: Demonstrates how to predict a single image. Update `test_image_path` to test with your own image.


## Results

The model achieves an accuracy of approximately 81% on the test set.

**Classification Report:**

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| glioma     | 0.78      | 0.82   | 0.80     | 288     |
| meningioma | 0.69      | 0.64   | 0.67     | 265     |
| notumor    | 0.86      | 0.84   | 0.85     | 291     |
| pituitary  | 0.88      | 0.93   | 0.90     | 299     |
|            |           |        |          |         |
| **Accuracy** |           |        | **0.81** | **1143**|
| **Macro Avg**| 0.80      | 0.81   | 0.80     | 1143    |
| **Weighted Avg**| 0.81      | 0.81   | 0.81     | 1143    |

**Accuracy Score:**

0.8101487314085739


These results indicate that the model performs reasonably well across all tumor categories, with particularly strong performance on 'pituitary' and 'notumor' classifications.

## Dependencies

-   `os`
-   `numpy`
-   `matplotlib`
-   `opencv-python` (cv2)
-   `scikit-learn` (sklearn)
-   `tensorflow` (for `ImageDataGenerator`, `img_to_array`, `load_img`)

## Contributing

Contributions are welcome! If you have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is open-source and available under the [MIT License](LICENSE).






