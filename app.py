import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from flask import Flask, request, jsonify, render_template
# Load the data from the CSV file
data = pd.read_csv('data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['language'], test_size=0.2, random_state=42)

# Define the model pipeline
model = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', RandomForestClassifier())
])

# Train the model on the training set
model.fit(X_train, y_train)

# Define the Flask app
app = Flask(__name__)

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for predicting the language of a given text
@app.route('/predict', methods=['POST'])
def predict():
    # get the text data from the request
    data = request.get_json()

    # process the text data to predict the language
    text = data['text']
    predicted_language = model.predict([text])[0]

    # return a JSON response with the predicted language
    return jsonify({'language': predicted_language})


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
