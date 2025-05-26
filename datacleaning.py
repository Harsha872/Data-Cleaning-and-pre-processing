import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df.head()

#explore the data
df.info()
df.describe()
df.isnull().sum()

#explore the data
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
df.isnull().sum()

#encode catgorical values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

#normalized features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_cols = ['Age', 'Fare']
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

#detect and visualize
# Boxplots for numeric features
plt.figure(figsize=(10,5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title('Boxplot for Age and Fare')
plt.show()

#remove outlers
# Remove rows where Fare > 3 std devs
from scipy import stats
df = df[(np.abs(stats.zscore(df[['Age', 'Fare']])) < 3).all(axis=1)]

#cleaned final data
df.head()
df.to_csv("cleaned_titanic.csv", index=False)






