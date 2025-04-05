Diabetes Prediction and Health Insights
=======================================

A machine learning project to predict diabetes risk using user health inputs.  
Provides clear predictions, health comparisons, and simple data visualizations.

Overview
--------

This Python-based tool uses a Support Vector Machine (SVM) to assess the likelihood of diabetes based on eight key health metrics.  
The output includes personalized insights and visual comparisons against dataset averages to promote better health awareness.

Features
--------

- Predicts diabetes risk using SVM
- Accepts user input via terminal
- Provides easy-to-understand feedback for each health metric
- Visualizes your health metrics vs population data
- Educational, simple, and lightweight

Health Metrics Used
-------------------

- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI (Body Mass Index)  
- Diabetes Pedigree Function  
- Age

Requirements
------------

Python 3.x and the following libraries:

```
numpy  
pandas  
matplotlib  
seaborn  
scikit-learn
```

Install all dependencies:

```
pip install numpy pandas matplotlib seaborn scikit-learn
```

How to Use
----------

1. Run the script:

```
python main.py
```

2. Enter your health data when prompted.
3. The program will display:
   - A diabetes risk prediction
   - Personalized feedback on each metric
   - Graphs showing how your values compare to the population

Files
-----

- main.py — Main script  
- diabetes.csv — Dataset used for training  
- README.md — Project documentation

Disclaimer
----------

This tool is for educational and informational purposes only.  
It is not a diagnostic tool. Please consult a licensed medical professional for health concerns.
