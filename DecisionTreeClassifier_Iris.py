from sklearn import tree
from sklearn import load_iris
iris = load_iris()
X, y = iris.data, iris.target
classy = tree.DecisionTreeClassifier()
classy = classy.fit(X, y)
tree.plot_tree(classy)
