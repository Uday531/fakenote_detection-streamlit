from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Project features to 1D for a clear decision boundary visualization
pca = PCA(n_components=1)
X1 = pca.fit_transform(X).flatten()

# Split the 1D data into train/test
X1_train, X1_test, y_train, y_test = train_test_split(
	X1, y, test_size=0.2, random_state=42, stratify=y
)

# Train Logistic Regression on the 1D projection
clf = LogisticRegression(max_iter=200)
clf.fit(X1_train.reshape(-1, 1), y_train)

# Evaluate on the 1D test set
y_pred = clf.predict(X1_test.reshape(-1, 1))
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy (1D projection): {acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Create a dense grid along the 1D axis and find where predicted class changes
x_grid = np.linspace(X1.min() - 1, X1.max() + 1, 2000).reshape(-1, 1)
pred_grid = clf.predict(x_grid)
changes = np.where(pred_grid[:-1] != pred_grid[1:])[0]
boundaries = []
for idx in changes:
	b = float((x_grid[idx] + x_grid[idx + 1]) / 2.0)
	boundaries.append(b)

# Plot true vs predicted along the 1D projection and vertical boundary lines
fig, ax = plt.subplots(figsize=(8, 3))
ax.scatter(X1_test, y_test, c=y_test, cmap='viridis', marker='o', edgecolors='k', label='True')
ax.scatter(X1_test, y_pred + 0.08, c=y_pred, cmap='cool', marker='x', label='Predicted')
for b in boundaries:
	ax.axvline(b, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('PCA component 1')
ax.set_yticks(range(len(data.target_names)))
ax.set_yticklabels(data.target_names)
ax.legend(loc='upper right')
plt.title('Logistic Regression decision boundaries on 1D projection')
plt.tight_layout()
plt.show()