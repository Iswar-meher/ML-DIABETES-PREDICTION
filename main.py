import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

# Load dataset
diabetes_dataset = pd.read_csv("diabetes.csv")

# Define features and target
features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
X = diabetes_dataset[features]
Y = diabetes_dataset["Outcome"]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

# Take user input safely
print("\n**Health Assessment: Enter Your Metrics**")
user_input = []
for f in features:
    while True:
        try:
            value = float(input(f"{f}: "))
            user_input.append(value)
            break
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

# Predict
user_input_np = np.asarray(user_input).reshape(1, -1)
user_input_scaled = scaler.transform(user_input_np)
prediction = classifier.predict(user_input_scaled)

# Display results
print("\n" + "="*40)
print("**Diabetes Risk Prediction**")
print("="*40)

if prediction[0] == 0:
    print("✅ **Great news! You are likely NOT diabetic.** 🎉")
    print("   Keep up a healthy lifestyle and regular checkups. 👍")
else:
    print("⚠️ **Caution! You may be at risk for diabetes.**")
    print("   Please consult a doctor for further tests. 🩺")
