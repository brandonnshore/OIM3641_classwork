# create_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

joblib.dump(clf, 'iris_model.joblib')
print('Model saved to iris_model.joblib')
