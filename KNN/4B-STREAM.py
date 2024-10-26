import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Load model and dataset
model = GaussianNB()
train_data = pd.read_csv(r'C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/KNN/Data Set (2)/SalaryData_Train.csv')
X_train = train_data.drop(columns=['Salary'])
y_train = train_data['Salary']

# Preprocessing
label_encoder = LabelEncoder()
for column in X_train.columns:
    if X_train[column].dtype == 'object':
        X_train[column] = label_encoder.fit_transform(X_train[column])

model.fit(X_train, y_train)

# Streamlit UI
st.title("Salary Prediction using Naive Bayes")

# Collect input from user
input_data = {}
for col in X_train.columns:
    input_data[col] = st.text_input(f"Enter {col}")

# Predict Salary
if st.button("Predict Salary"):
    input_df = pd.DataFrame([input_data])
    for col in input_df.columns:
        input_df[col] = label_encoder.transform(input_df[col])
    prediction = model.predict(input_df)
    st.write(f"Predicted Salary: {prediction[0]}")
