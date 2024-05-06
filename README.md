# Insurance Claim Prediction Documentation

## Overview

This project aims to predict insurance claim amounts based on various factors such as age, BMI, blood pressure, number of children, gender, diabetic status, smoker status, and region. Predicting insurance claim amounts can assist insurance companies in understanding their potential liabilities and help them make informed decisions about risk management and pricing.

## Project Components

### 1. Data Loading and Preprocessing

- **Data Source:** The dataset used for this project is sourced from [provide source details].
- **Preprocessing Steps:**
  - Handling missing values: Missing values in the 'age' and 'region' columns are replaced with the median age and mode region, respectively.
  - One-hot encoding: Categorical variables ('gender', 'diabetic', 'smoker', 'region') are converted into binary features using one-hot encoding.

### 2. Model Building

- **Models Used:** 
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
  - Support Vector Regression

- **Evaluation:** Mean Absolute Error (MAE) is used as the evaluation metric to assess the performance of each model on the test set.

### 3. Streamlit Application

- **Functionality:** The Streamlit application allows users to input their information (age, BMI, blood pressure, number of children, gender, diabetic status, smoker status, and region) and obtain a prediction of their insurance claim amount.
- **Pipeline:** A pipeline is used to preprocess the input data and make predictions using the trained Random Forest Regressor model.
- **User Interface:** The application provides a user-friendly interface with sliders, input fields, and dropdown menus for easy input of user information.

## Usage

To use the application:
1. Run the Streamlit application script (`streamlit_app.py`).
2. Input your information in the provided fields.
3. Click on the "Predict" button to obtain the predicted insurance claim amount.

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- streamlit

## Future Improvements

- Include additional features or external data sources for better prediction accuracy.
- Implement feature selection techniques to identify the most important predictors.
- Enhance the user interface with additional features and visualizations.

## Contributors

- Mr. SAAD AHMED MASOOD
- Mr. MESUM RAZA
