
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Train Linear SVM
linear_svm = SVC(kernel="linear", C=1.0)
linear_svm.fit(X_train, y_train)

# Train RBF SVM
rbf_svm = SVC(kernel="rbf", C=1.0, gamma="scale")
rbf_svm.fit(X_train, y_train)

# Predictions
linear_preds = linear_svm.predict(X_test)
rbf_preds = rbf_svm.predict(X_test)

# Accuracy Comparison
print("Linear SVM Accuracy:", accuracy_score(y_test, linear_preds))
print("RBF SVM Accuracy:", accuracy_score(y_test, rbf_preds))

import matplotlib.pyplot as plt
import seaborn as sns

# Select two features for visualization
X_vis = X_scaled[:, [0, 1]]  # Use first two features
X_vis_train, X_vis_test = train_test_split(X_vis, test_size=0.2, random_state=42)

# Train SVM on two features
svm_vis = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_vis.fit(X_vis_train, y_train)

# Decision Boundary Plot
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_vis_test, y_test, clf=svm_vis, legend=2)

plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title("SVM Decision Boundary (RBF Kernel)")
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1, 1]}
grid_search = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores))


