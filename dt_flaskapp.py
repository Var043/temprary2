
# Creating a new Flask application and importing the necessary libraries, 
# such as NumPy and Scikit-learn (if you are using a pre-trained model)

from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


# Create an instance of the Flask class and define routes for the web page.
app = Flask(__name__)
model=pickle.load(open('DT_model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')


# Creating another route that will handle the form submission and use the model to make a prediction

@app.route('/predict', methods=['POST'])
def predict():
    SepalLengthCm = float(request.form['SepalLengthCm'])
    SepalWidthCm = float(request.form['SepalWidthCm'])
    PetalLengthCm = float(request.form['PetalLengthCm'])
    PetalWidthCm = float(request.form['PetalWidthCm'])
    features = np.array([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]).reshape(1, -1)
    probability=model.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
    return render_template('predict.html', probability=probability)


# Finally, run the application using
if __name__ == '__main__':
    app.run(debug=True)
