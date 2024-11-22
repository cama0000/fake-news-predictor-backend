from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import re
import nltk
from nltk.corpus import stopwords
import unicodedata
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from bs4 import BeautifulSoup
import requests
import joblib
from flask_cors import CORS

# for cleaning text
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() 
        url = data['url']
        model_type = data['model']

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        if model_type == 'cnn':
            model = joblib.load('cnn_model.sav')

            with open('cnn_tokenizer.pkl', 'rb') as file:
                tokenizer = pickle.load(file)
        elif model_type == 'lstm':
            model = joblib.load('lstm_model.sav')

            with open('lstm_tokenizer.pkl', 'rb') as file:
                tokenizer = pickle.load(file)

        max_sequence_length = 200

        raw_text = web_scrape(url)
        article_text = clean_text(raw_text)

        article_seq = tokenizer.texts_to_sequences([article_text])
        article_padded = pad_sequences(article_seq, maxlen=max_sequence_length, padding='post')

        prediction = model.predict(article_padded)
    
        prediction_value = prediction[0][0] 
        
        return jsonify({'prediction': float(prediction_value)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def web_scrape(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        headline = soup.find('h1')
        if headline:
            headline_text = headline.get_text().strip()
        else:
            headline_text = "Headline not found"

        # Find the paragraphs using specific classes or tags
        article_paragraphs = soup.find_all('div', class_='article-body')
        if not article_paragraphs:  # If not found, try another method
            article_paragraphs = soup.find_all('p')  # General search for paragraphs
        if article_paragraphs:
            article_text = ' '.join([p.get_text().strip() for p in article_paragraphs])
        else:
            article_text = "Article content not found"

        full_article = headline_text + ' ' + article_text

        print(full_article)
        return full_article
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")

 

# DATA PRE-PROCESSING (Stopwords, HTML tags, punctuation, makes lowercase)
def clean_text(text):
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())

    # remove accents
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    # remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

@app.route("/")
def hello_world():
    return "You have accessed the hello world endpoint! That means the server is working!"

if __name__ == '__main__':
    app.run(debug=True)

