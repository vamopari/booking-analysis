import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

from preprocessing import load_clean_data, split_data, build_preprocessor

df = load_clean_data()
X_train, X_test, y_train, y_test = split_data(df)

preprocessor = build_preprocessor()
model = KNeighborsClassifier(n_neighbors=15)

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", model),
])

start = time.time()
pipe.fit(X_train, y_train)
fit_time = time.time() - start

start = time.time()
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]
predict_time = time.time() - start

print(f"Fit time:     {fit_time:.2f}s")
print(f"Predict time: {predict_time:.2f}s")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("AUC:      ", roc_auc_score(y_test, y_proba))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:   ", recall_score(y_test, y_pred))
print("F1:       ", f1_score(y_test, y_pred))
print("MCC:      ", matthews_corrcoef(y_test, y_pred))
