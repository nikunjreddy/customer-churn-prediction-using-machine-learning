# Telco Customer Churn Prediction Project

## Overview
This project aims to compare various models for predicting customer churn in a telecommunications company. 
The dataset used in this project is loaded from 'data.csv' and includes information about customer attributes and whether they churned or not.
The Dataset contains 7043 rows and 21 columns with the main variable as churn.

## Project Structure
1. **Data Exploration:**
    - Loaded and explored the dataset using Pandas.
    - Checked data shape, types, and basic statistics.
    - Investigated missing values and visualized data distributions.

2. **Data Visualization:**
    - Utilized histograms and bar plots to visualize the distribution of numerical and categorical features.
    - Created a correlation heatmap to examine relationships between numerical features.
    - Employed box plots to observe how numerical features vary for customers who churned and those who did not.
    - Presented a pie chart to illustrate the distribution of churn in the dataset.

3. **Data Preprocessing:**
    - Encoded categorical variables using Label Encoding.
    - Split the data into training and testing sets.
    - Standardized numerical features using Standard Scaler.
    - Encoded the target variable (Churn) using Label Encoding.

4. **Model Implementation:**
    - Implemented the following machine learning models:
        - Random Forest Classifier
        - Artificial Neural Network (ANN)
        - K-Nearest Neighbors (KNN)
        - Logistic Regression
        - Gaussian Naive Bayes
        - Decision Tree
        - Support Vector Machine (SVM)
        - Linear Discriminant Analysis (LDA)
        - XGBoost

5. **Model Evaluation:**
    - Trained and evaluated each model on the test set.
    - Calculated accuracy, confusion matrix, and classification report for each model.

6. **Model Comparison:**
    - Compiled a list of model names and their corresponding accuracies.
    - Visualized the accuracy of each model using a bar plot.

## Libraries Used

- Pandas
- NumPy
- Scikit-Learn
- TensorFlow
- Keras
- XGBoost
- Matplotlib
- Seaborn

## How to Run the Code

1. Ensure you have the necessary Python libraries installed using `pip install -r requirements.txt`.

2. Open and run the Jupyter Notebook `Telco_Churn_Prediction.ipynb` in a Jupyter environment.

## Results
The following table presents the accuracy of each model:

| Model                        | Accuracy |
|------------------------------|----------|
| Random Forest                | 0.80     |
| Artificial Neural Network    | 0.80     |
| K-Nearest Neighbors          | 0.77     |
| Logistic Regression          | 0.81     |
| Gaussian Naive Bayes         | 0.76     |
| Decision Tree                | 0.72     |
| Support Vector Machine       | 0.82     |
| Linear Discriminant Analysis | 0.82     |
| XGBoost                      | 0.79     |

## Conclusion
- Support Vector Machine (SVM) and Linear Discriminant Analysis (LDA) achieved the highest accuracy among the models.
- Logistic Regression and Artificial Neural Network also performed well.
