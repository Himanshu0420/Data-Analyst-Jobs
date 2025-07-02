import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

data = pd.read_csv('DataAnalyst.csv')
print(data.head())
print(data.info())
print(data.describe(include='all'))

print(f"Duplicate rows: {data.duplicated().sum()}")
data.drop_duplicates(inplace=True)
data['Rating'].fillna(data['Rating'].median(), inplace=True)
data.dropna(thresh=len(data)*0.7, axis=1, inplace=True)
categorical_cols = ['Company Name', 'Industry', 'Sector', 'Type of ownership']
data[categorical_cols] = data[categorical_cols].fillna(method='ffill')

data['Min Salary'] = data['Salary Estimate'].str.extract(r'(\d+)').astype(float)
data['Max Salary'] = data['Salary Estimate'].str.extract(r'-\s*(\d+)').astype(float)
data['Avg Salary'] = (data['Min Salary'] + data['Max Salary']) / 2
data.drop('Salary Estimate', axis=1, inplace=True)

data['Python'] = data['Job Description'].str.contains('Python', case=False).astype(int)
data['Excel'] = data['Job Description'].str.contains('Excel', case=False).astype(int)
data['Tech_Skills'] = data['Python'] + data['Excel']
data['City'] = data['Location'].str.split(',', expand=True)[0]
data['State'] = data['Location'].str.split(',', expand=True)[1]

sns.boxenplot(data['Avg Salary'], kde=True, bins=20)
plt.title("Salary Estimate Distribution")
plt.xlabel("Salary")
plt.show()

sns.boxplot(x='Industry', y='Rating', data=data)
plt.xticks(rotation=90)
plt.title("Company Ratings by Industry")
plt.show()

features = ['Rating', 'Tech_Skills']
X = data[features]
y = data['Avg Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)

plot_data = data[['Industry', 'Rating']].dropna()

# Plot with cleaned data
plt.figure(figsize=(12, 6))
sns.boxplot(x='Industry', y='Rating', data=plot_data)
plt.xticks(rotation=90)
plt.title("Company Ratings by Industry")
plt.tight_layout()
plt.show()

data_model = data[['Rating', 'Tech_Skills', 'Avg Salary']].dropna()

X = data_model[['Rating', 'Tech_Skills']]
y = data_model['Avg Salary']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

import joblib
joblib.dump(model, 'salary_predictor.pkl')
