from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request
import joblib
app = Flask(__name__)

model = joblib.load('sentiment_analysis_model.joblib')

vectorizer = joblib.load('vectorizer.joblib')

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']

        text_vectorized = vectorizer.transform([user_input])

        prediction = model.predict(text_vectorized)[0]

        return render_template('result.html', user_input=user_input, sentiment= prediction )
    
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])

    prediction = model.predict(text_vectorized)[0]

    return prediction

if __name__ == '__main__':
    app.run(debug=True)