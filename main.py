import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

# Load dataset
diabetes_dataset = pd.read_csv("diabetes.csv")

print(diabetes_dataset.head)