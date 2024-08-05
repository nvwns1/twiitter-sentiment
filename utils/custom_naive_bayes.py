import numpy as np
import joblib
import pandas as pd
from utils.functions import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report


class CustomNaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}
        self.classes = None
        self.vocab = None

    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier according to the training data.

        Parameters:
        X: sparse matrix, shape (n_samples, n_features)
           Training data
        y: array-like, shape (n_samples,)
           Target labels
        """
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.vocab = np.arange(X.shape[1])  # Feature indices

        # Initialize probabilities
        self.feature_probs = {c: np.zeros(len(self.vocab)) for c in self.classes}
        self.class_probs = {
            c: count / len(y) for c, count in zip(self.classes, class_counts)
        }

        # Count feature occurrences for each class
        for c in self.classes:
            class_data = X[y == c]
            word_counts = np.array(class_data.sum(axis=0)).flatten()
            self.feature_probs[c] = (word_counts + 1) / (
                word_counts.sum() + len(self.vocab)
            )  # Add-one smoothing

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters:
        X: sparse matrix, shape (n_samples, n_features)
           Test data

        Returns:
        array, shape (n_samples,)
            Predicted target values
        """
        predictions = []
        for doc in X:
            log_probs = {c: np.log(prob) for c, prob in self.class_probs.items()}
            word_counts = np.array(doc.sum(axis=0)).flatten()
            for c in self.classes:
                log_probs[c] += np.sum(np.log(self.feature_probs[c]) * word_counts)
            predictions.append(max(log_probs, key=log_probs.get))
        return np.array(predictions)


def train_model():
    df = pd.read_csv("dataset/TwitterDataset.csv", encoding="latin1")
    df.drop(columns=df.columns[0], inplace=True)
    # Renaming the 'Caption' to 'tweet'
    df.rename(columns={"Caption": "tweet"}, inplace=True)
    df.rename(columns={"LABEL": "sentiment"}, inplace=True)

    label_mapping = {"negative": 0, "neutral": 2, "positive": 4}
    df["sentiment"] = df["sentiment"].map(label_mapping)

    df["cleaned_tweets"] = df["tweet"].apply(preprocess_text)

    # Feature Extraction
    X = df["cleaned_tweets"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = CustomNaiveBayes()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
    joblib.dump(model, "model/custom.pkl")
