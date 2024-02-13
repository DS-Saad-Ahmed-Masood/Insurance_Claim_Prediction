#Step-1 Data Exploration and Preprocessing/ EDA
import pandas as pd

# Load the data
#df=pd.read_csv(r'C:\Users\SAM HP\Desktop\DataSet\insurance_data.csv')
df=pd.read_csv('insurance_data.csv')
print("First few rows of the dataframe:")
print(df.head())


print("\nData types of columns:")
print(df.dtypes)

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

#Step-2 Feature Engineering
# Handle missing values
median_age = df['age'].median()
df['age'].fillna(median_age, inplace=True)

mode_region = df['region'].mode()[0]
df['region'].fillna(mode_region, inplace=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['gender', 'diabetic', 'smoker'], drop_first=True)
df = pd.get_dummies(df, columns=['region'], drop_first=False)  # Preserve all unique values in 'region'


#print(df.head())

#print("\nMissing values after preprocessing:")
print(df.isnull().sum())

#Step-3 Model Building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['claim'])
y = df['claim']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the models
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

svr_model = SVR()
svr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_linear = linear_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_svr = svr_model.predict(X_test)

# Evaluate the models
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mae_svr = mean_absolute_error(y_test, y_pred_svr)

print(f'Linear Regression: Mean Absolute Error = {mae_linear}')
print(f'Random Forest Regressor: Mean Absolute Error = {mae_rf}')
print(f'Gradient Boosting Regressor: Mean Absolute Error = {mae_gb}')
print(f'XGBoost Regressor: Mean Absolute Error = {mae_xgb}')
print(f'Support Vector Regression: Mean Absolute Error = {mae_svr}')

#The Mean Absolute Error (MAE) value you obtained is a metric used to evaluate the performance of a regression model. It represents the average absolute difference between the predicted values and the actual values in the test set. 
#The MAE measures the average magnitude of errors in a set of predictions, without considering their direction (whether they are overestimations or underestimations).

#Step 4 Streamlit APP
import streamlit as st
import numpy as np

# Load the trained Random Forest Regressor model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)


# Define the Streamlit app
def main():    
    st.set_page_config(
    page_title="Insurance Claim Prediction",
    page_icon="⚕️",layout="wide"
    )
    st.title('Insurance Claim Prediction')
    
    with st.container():
        col1,col2= st.columns(spec=[0.5,0.5], gap="small")
    
    with col1:
        st.image("https://thumbor.forbes.com/thumbor/fit-in/900x510/https://www.forbes.com/advisor/in/wp-content/uploads/2022/11/health-insurance-image-scaled.jpg",use_column_width=True)
        st.write("Welcome to Insurance Claim Prediction App! This application is designed to assist you in estimating the expected insurance claim amount based on various factors. Whether you're an insurance agent, policyholder, or anyone interested in insurance analytics, this tool provides valuable insights into potential claim amounts")
        
        st.subheader("How It Works:")
        st.write("**Input Your Information:** Enter your age, BMI (Body Mass Index), blood pressure, number of children, gender, diabetic status, smoker status, and region.")
        st.write("**Get Your Prediction:** Advanced machine learning models analyze the provided information to predict the expected insurance claim amount.")
        
        st.subheader("Key Features:")
        st.write("**Predictive Models:** Employ sophisticated machine learning algorithms, including Linear Regression, Random Forest Regression, Gradient Boosting Regression, XGBoost Regression, and Support Vector Regression, to make accurate predictions.")
        st.write("**User-Friendly Interface:** The streamlined interface makes it easy to input your data and receive instant predictions.")
        st.write("**Comprehensive Insights:** Gain valuable insights into factors influencing insurance claim amounts and make informed decisions.")
        
        st.subheader("Benefit to use this APP:")
        st.write("**Efficiency:** Save time and resources by obtaining quick and accurate insurance claim predictions.")
        st.write("**Decision Support:** Whether you're an insurance professional assessing risk or an individual planning for future expenses,this app provides valuable guidance.")
        st.write("**Accessibility:** Accessible from any device with an internet connection, ensuring convenience wherever you are.")
        st.info("Disclaimer:Please note that the predictions provided by this application are based on statistical models and may not represent actual claim amounts. Always consult with a qualified insurance professional for personalized advice.")
        st.success("Created by:**Mr. Saad Ahmed Masood**")
        
    with col2:       
    # Collect user inputs
        age = st.slider('**Age**', min_value=18, max_value=120, value=35)
        bmi = st.number_input('**BMI**', min_value=10.0, max_value=50.0, value=30.0)
        blood_pressure = st.number_input('**Blood Pressure**', min_value=50, max_value=200, value=120)
        children = st.number_input('**Number of Children**', min_value=0, max_value=10, value=0)
        gender = st.radio('**Gender**', ['Male', 'Female'])
        diabetic = st.radio('**Diabetic**', ['Yes', 'No'])
        smoker = st.radio('**Smoker**', ['Yes', 'No'])
        region = st.selectbox('**Region**', ['Northeast', 'Northwest', 'Southeast', 'Southwest'])

    # Preprocess user inputs
        gender = 0 if gender == 'Male' else 1
        diabetic = 1 if diabetic == 'Yes' else 0
        smoker = 1 if smoker == 'Yes' else 0
        region_mapping = {'Northeast': 0, 'Northwest': 1, 'Southeast': 2, 'Southwest': 3}
        region = region_mapping[region]

    # Make prediction
        if st.button('Predict Claim Amount'):
            prediction = predict_claim_amount(age, bmi, blood_pressure, children, gender, diabetic, smoker, region)
            st.success(f'Predicted Insurance Claim Amount: ${prediction:.2f}')
        
    # Display prediction
    #st.write('Predicted Insurance Claim Amount:', prediction)


# Function to make prediction using the trained model
def predict_claim_amount(age, bmi, blood_pressure, children, gender, diabetic, smoker, region):
    # Preprocess user inputs
    gender = 1 if gender == 'male' else 0
    diabetic = 1 if diabetic == 'Yes' else 0
    smoker = 1 if smoker == 'Yes' else 0
    region_mapping = {'northwest': 0, 'southeast': 1, 'southwest': 2}
    region_str = str(region)  # Convert to string to ensure it has the strip and lower methods
    region_str = region_str.strip().lower()  # Convert to lowercase and remove leading/trailing whitespace
    region = region_mapping.get(region_str, 0)  # Use get method to handle KeyError

    # Create feature array
    features = np.array([[age, bmi, blood_pressure, children, gender, diabetic, smoker,
                          1 if region == 0 else 0, 1 if region == 1 else 0, 1 if region == 2 else 0,
                          0, 0, 0]])  # Add placeholder columns for missing features

    # Make prediction
    predicted_claim_amount = rf_model.predict(features)
    return predicted_claim_amount[0]

if __name__ == '__main__':
    main()
