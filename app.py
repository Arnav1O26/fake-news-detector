from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        
        # Transform the input text using the trained vectorizer
        text_vectorized = vectorizer.transform([news_text])
        
        # Make prediction (0 for Fake, 1 for Real)
        prediction = model.predict(text_vectorized)[0]
        
        result = "Real News" if prediction == 1 else "Fake News"
        
        return render_template('index.html', prediction_text=f'Prediction: {result}', original_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)