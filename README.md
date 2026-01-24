# ML_assignment2
 Implement multiple classification models -  Build an interactive Streamlit web application to demonstrate your models - Deploy  the app on Streamlit Community Cloud

## Problem Statement
Early detection of breast cancer is crucial for improving patient survival rates.
This project aims to build and compare multiple machine learning classification
models to predict whether a breast tumor is **Malignant** or **Benign** based on
diagnostic features extracted from medical images.

## Dataset Description
The Breast Cancer Wisconsin (Diagnostic) dataset was obtained from the
UCI Machine Learning Repository.

- Total Instances: 569
- Total Features: 30 numerical features
- Target Variable: Diagnosis
  - M → Malignant
  - B → Benign

The dataset contains computed features derived from digitized images of
fine needle aspirate (FNA) of breast masses.

### Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|-----|-----------|--------|----------|-----|
|Logistic Regression |0.964912|0.996032|0.975|0.928571|0.95122|0.924518|
|Decision Tree |0.929825|0.924603|0.904762|0.904762|0.904762|0.849206|
|KNN |0.95614|0.982308|0.974359|0.904762|0.938272|0.905824|
|Naive Bayes |0.938596|0.993386|1|0.833333|0.909091|0.871489|
|Random Forest |0.973684|0.99289|1|0.928571|0.962963|0.944155|
|XGBoost |0.973684|0.994048|1|0.928571|0.962963|0.944155|


## Observations

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Performed very well due to the linear separability of the dataset and achieved high precision and recall. |
| Decision Tree | Showed lower generalization performance and was prone to overfitting compared to ensemble methods. |
| kNN | Achieved strong performance but was sensitive to feature scaling and choice of k value. |
| Naive Bayes | Performed reasonably well despite its strong independence assumption among features. |
| Random Forest (Ensemble) | Improved overall performance by reducing variance through ensemble learning. |
| XGBoost (Ensemble) | Achieved the best performance across most metrics due to gradient boosting and effective handling of feature interactions. |

