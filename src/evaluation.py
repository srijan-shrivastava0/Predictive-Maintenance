
from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    print("Classification Report:\n", classification_report(y_test, preds))
    print("ROC AUC Score:", roc_auc_score(y_test, probas))
