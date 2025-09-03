# Rice Variety Classification

Inspired by ["Identification of Rice Varieties Using Machine Learning Algorithms"](https://doi.org/10.15832/ankutbd.862482) (Cinar & Koklu, 2022).  
This project was completed as part of a **Data Science module assignment**. The goal was to extract features from rice grain images, visualize them using PCA, and train machine learning models to classify rice varieties.

## 📂 Project Structure
- feature_extraction.py # Extract shape & RGB features from images
- visualize_contours.py # Visualize contours and fitted ellipses for a sample image
- pca_visualization.py # PCA plots with and without labels
- train_knn.py # K-Nearest Neighbors training with hyperparameter tuning
- train_logistic_regression.py # Logistic Regression training with hyperparameter tuning
- train_svm.py # SVM training with hyperparameter tuning
- requirements.txt # Required Python libraries
- .gitignore # Ignore dataset and generated CSV
- README.md

- **Rice_Image_Dataset/** and **rice_features.csv** are ignored in `.gitignore` because they are large/generated files.  

## 🖼 Dataset

- The dataset contains five rice varieties:  
  `Arborio`, `Basmati`, `Ipsala`, `Jasmine`, and `Karacadag`.  
- Each image contains a single rice grain with a distinguishable background.  

> **Note:** The dataset is not included in the repository. You need to add your dataset folder as `Rice_Image_Dataset/` to run the scripts. The dataset is accessible here: https://www.muratkoklu.com/datasets/

## ⚙️ Features Extracted

- **Shape features:** `area`, `perimeter`, `major_axis`, `minor_axis`, `aspect_ratio`, `eccentricity`, `solidity`, `roundness`  
- **Color features:** mean and standard deviation of `R`, `G`, `B` channels  

## 📊 PCA Visualization

- `pca_visualization.py` produces:
  - PCA plot **with labels** to see class separation.
  - PCA plot **without labels** to visualize overall clustering.

## 🤖 Machine Learning Models

- **K-Nearest Neighbors (KNN)** – hyperparameter tuning with GridSearchCV  
- **Logistic Regression** – hyperparameter tuning for penalty and C  
- **Support Vector Machine (SVM)** – hyperparameter tuning for kernel, C, gamma  

Scripts automatically standardize features and evaluate models using accuracy, classification report, and confusion matrix.
