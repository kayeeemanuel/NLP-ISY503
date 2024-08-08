# model.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
        return "Positive Review"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative Review"
    else:
        return "Neutral Review"
