
import pandas as pd

url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
df = pd.read_csv(url)

print(df.info())
print(df.head())
# Basic statistical summary
print(df.describe())

# Checking missing values
print(df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram for 'Age'
plt.figure(figsize=(6,4))
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Boxplot for 'Fare'
plt.figure(figsize=(6,4))
sns.boxplot(x=df['Fare'])
plt.title("Fare Distribution")
plt.show()

# Pairplot for selected features
sns.pairplot(df[['Age', 'Fare', 'Survived']])
plt.show()

# Correlation matrix
plt.figure(figsize=(8,6))
sns.heatmap(df[['Age', 'Fare', 'Survived']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# Detecting skewness
print(df.skew())

# Checking multicollinearity
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Full Feature Correlation")
plt.show()
