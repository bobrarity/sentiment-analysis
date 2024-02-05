# Importing the libs
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Word tokenize
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens_filtered = [word.lower() for word in tokens if
                       word.isalpha() and word.lower() not in stopwords.words('english')]
    return " ".join(tokens_filtered)


# Text stemming
def stem_text(text):
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in word_tokenize(text.lower())])


# Load and preprocess training data
train_df = pd.read_csv('/usr/src/app/data/raw/train.csv')
train_df['processed_review'] = train_df['review'].apply(preprocess_text)
train_df['stemmed_review'] = train_df['processed_review'].apply(stem_text)

# Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['stemmed_review'])
y_train = train_df['sentiment'].map({'positive': 1, 'negative': 0})

print('Train data has been preprocessed.')

# Model Training with GridSearchCV
linear_svc_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'dual': [False]
}
linear_svc_model = LinearSVC(max_iter=10000)
linear_svc_grid_search = GridSearchCV(linear_svc_model, linear_svc_param_grid, cv=5, error_score='raise')
linear_svc_grid_search.fit(X_train_tfidf, y_train)

print('LinearSVC model has been trained.')

# Save the trained model
joblib.dump(linear_svc_grid_search.best_estimator_, '/usr/src/app/outputs/models/sentiment_model.pkl')

print('LinearSVC model saved.')

# Preprocess and vectorize test data
test_df = pd.read_csv('/usr/src/app/data/raw/test.csv')
test_df['processed_review'] = test_df['review'].apply(preprocess_text)
test_df['stemmed_review'] = test_df['processed_review'].apply(stem_text)

# Save preprocessed test data, (so, we don't have to do it in inference file)
X_test_tfidf = tfidf_vectorizer.transform(test_df['stemmed_review'])
joblib.dump(X_test_tfidf, '/usr/src/app/data/preprocessed_test/X_test_tfidf.pkl')

y_test = test_df['sentiment'].map({'positive': 1, 'negative': 0})
joblib.dump(y_test, '/usr/src/app/data/preprocessed_test/y_test.pkl')

print('Preprocessed test data saved.')
