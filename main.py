#import dependencies
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
user_input_df = pd.DataFrame([user_input], columns=features)# Wrap user input with feature names
user_input_scaled = scaler.transform(user_input_df)# Standardize using scaler
prediction = classifier.predict(user_input_scaled)

# Display results
print("\n" + "="*40)
print("**Diabetes Risk Prediction**")
print("="*40)

if prediction[0] == 0:
    print("✅ **Great news! You are likely NOT diabetic.** ")
    print("   Keep up a healthy lifestyle and regular checkups.\n\n")
else:
    print("⚠️ **Caution! You may be at risk for diabetes.**")
    print("   Please consult a doctor for further tests. \n\n")

# Compare to dataset averages
df_mean = diabetes_dataset[features].mean()
for i, f in enumerate(features):
    user_val = user_input[i]
    avg_val = df_mean[f]
    
    if user_val > avg_val * 1.2:
        status = "⬆️ HIGHER than average"
    elif user_val < avg_val * 0.8:
        status = "⬇️ LOWER than average"
    else:
        status = "✔️ NORMAL range"
    
    print(f"🔹 {f}: {user_val} (Avg: {avg_val:.2f}) → {status}")

print("-" * 40)
print("\n**Tip:** High glucose, BMI, or insulin levels may increase diabetes risk. \n   Consult a healthcare provider for personalized advice. 💙")

# Visualization
print("\n**Understanding Your Health Metrics**")
print("🔹 The graphs below compare your values with the general population.\n"
      "🔹 **Red Line** ➝ Your value\n"
      "🔹 **Blue Curve** ➝ Distribution of people in dataset")

plt.figure(figsize=(15, 10))

for i, f in enumerate(features):
    plt.subplot(3, 3, i + 1)
    
    # Create histogram with KDE (density curve)
    sns.histplot(diabetes_dataset[f], kde=True, bins=30, color='skyblue', alpha=0.6)
    
    # Mark user value with a red vertical line
    plt.axvline(user_input[i], color='red', linestyle='--', linewidth=2, label="Your Value")
    
    # Title with clear context
    plt.title(f"{f} (Your Value: {user_input[i]:.2f})", fontsize=12, fontweight='bold')
    
    # Remove grid lines for better readability
    plt.grid(False)
    plt.legend()

plt.tight_layout()
plt.show()

print("\n **How to Read the Graphs?**")
print("   - If your red line is far from the peak, you may have an unusual value.")
print("   - If it's near the center, your value is within the common range.")
print("   - Higher or lower values don't always mean bad health, but can be useful to discuss with a doctor.")

