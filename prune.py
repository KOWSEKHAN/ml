from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train full tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Prune tree using optimal alpha
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alpha = path.ccp_alphas[np.argmax([DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
                                       .fit(X_train, y_train).score(X_test, y_test) for alpha in path.ccp_alphas])]
pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha).fit(X_train, y_train)

# Plot pruned tree
plt.figure(figsize=(12, 8))
plot_tree(pruned_tree, filled=True)
plt.title("Pruned Decision Tree")
plt.show()