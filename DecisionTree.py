import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv('Exp 10.csv')
X = pd.get_dummies(data[['Age','Income','Credit Rating']])
y = data['Loan Approved'].map({'Yes':1,'No':0})

model = DecisionTreeClassifier(criterion="entropy",max_depth=3)
model.fit(X,y)

plt.figure(figsize=(12,8))
plot_tree(model, feature_names=X.columns, class_names=['No','Yes'],filled=True)
plt.savefig("decision_tree_output.png")
plt.show()