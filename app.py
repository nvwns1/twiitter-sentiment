from flask import Flask, request, render_template, jsonify
from utils.functions import predict_tweets
from utils.custom_naive_bayes import train_model

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    predictedLabel = ""
    formatted_probs = ""
    data = request.get_json()
    tweet = data.get("tweets", "")
    model_type = data.get("type", "")

    predictedLabel, formatted_probs = predict_tweets(tweet, model_type)
    return jsonify(
        {
            "model_type": model_type,
            "predictedLabel": predictedLabel,
            "formatted_probs": formatted_probs,
        }
    )


@app.route("/train")
def train():
    train_model()
    return jsonify({"message": "Model Trained Successfully"})


if __name__ == "__main__":
    app.run(debug=True)
