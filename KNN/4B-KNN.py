import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the datasets
test_data = pd.read_csv(r"C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/KNN/Data Set (2)/SalaryData_Test.csv")
train_data = pd.read_csv(r"C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/KNN/Data Set (2)/SalaryData_Train.csv")

# Separate features and target
X_train = train_data.drop(columns=['Salary'])
y_train = train_data['Salary']

X_test = test_data.drop(columns=['Salary'])
y_test = test_data['Salary']

# Encode categorical variables
label_encoder = LabelEncoder()
for column in X_train.columns:
    if X_train[column].dtype == 'object':
        X_train[column] = label_encoder.fit_transform(X_train[column])
        X_test[column] = label_encoder.transform(X_test[column])

# Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predictions
y_pred = nb_model.predict(X_test)

# Model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
