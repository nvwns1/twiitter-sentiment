import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# from utils.custom_naive_bayes import CustomNaiveBayes

label_mapping = {0: "negative", 2: "neutral", 4: "positive"}


# Removing @ from the tweet
def remove_usernames(text):
    return re.sub(r"@\w+", "", text)


def remove_hashtags(text):
    return re.sub(r"#", "", text)


def clean_text(text):
    # Removing URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Removing special characters and numbers
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\d", "", text)
    # Removing extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text):
    return word_tokenize(text)


def case_folding(text):
    return text.lower()


def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words]


def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]


def preprocess_text(text):
    text = remove_usernames(text)
    text = remove_hashtags(text)
    text = clean_text(text)
    text = case_folding(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return " ".join(tokens)


model1 = joblib.load("model/model.pkl")
vectorizer1 = joblib.load("model/vectorizer.pkl")

svmModel = joblib.load("model/svm.pkl")

customNB = joblib.load("model/custom.pkl")


def predict_tweets(new_tweets, model_type):
    new_tweets_cleaned = [preprocess_text(new_tweets)]

    new_tweets_tfidf = vectorizer1.transform(new_tweets_cleaned)

    if model_type == "NaiveBayes":
        predictions = model1.predict(new_tweets_tfidf)
        predictedLabel = label_mapping[predictions[0]]
        predictedProb = model1.predict_proba(new_tweets_tfidf)

        # Mapping predicted probabilities to labels
        prob_dict = {
            label: prob * 100
            for label, prob in zip(label_mapping.values(), predictedProb[0])
        }

        # Format the probabilities as percentages
        formatted_probs = {label: f"{prob:.0f}%" for label, prob in prob_dict.items()}

        return predictedLabel, formatted_probs

    if model_type == "SVM":
        # formatted_probs = ""
        predictions = svmModel.predict(new_tweets_tfidf)
        predictedLabel = label_mapping[predictions[0]]

        predictedProb = svmModel.predict_proba(new_tweets_tfidf)
        print(predictedProb)
        # # Mapping predicted probabilities to labels
        prob_dict = {
            label: prob * 100
            for label, prob in zip(label_mapping.values(), predictedProb[0])
        }

        # # Format the probabilities as percentages
        formatted_probs = {label: f"{prob:.0f}%" for label, prob in prob_dict.items()}

        return predictedLabel, formatted_probs

    if model_type == "NaiveBayesHardCoded":
        predictions = customNB.predict(new_tweets_tfidf)
        formatted_probs = ""
        predictedLabel = label_mapping[predictions[0]]
        return predictedLabel, formatted_probs

    return "Invalid model type selected."
