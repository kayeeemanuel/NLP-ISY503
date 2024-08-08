# app.py
from flask import Flask, request, render_template
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re

nltk.download('vader_lexicon')

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

# Function to parse the dataset
def parse_reviews(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    reviews = re.findall(r'<review>(.*?)</review>', content, re.DOTALL)
    return reviews

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for analyzing sentiment
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment = sia.polarity_scores(text)
    sentiment_score = sentiment['compound']
    sentiment_label = 'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral'
    return render_template('index.html', sentiment=sentiment_label, text=text)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
