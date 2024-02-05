import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

model = joblib.load('/usr/src/app/outputs/models/sentiment_model.pkl')

X_test_tfidf = joblib.load('/usr/src/app/data/preprocessed_test/X_test_tfidf.pkl')

y_test = joblib.load('/usr/src/app/data/preprocessed_test/y_test.pkl')

y_pred = model.predict(X_test_tfidf)

predictions_df = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
predictions_df.to_csv('/usr/src/app/outputs/predictions/predictions.csv', index=False)

print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')
print(f'F1 score: {f1_score(y_test, y_pred)}')
