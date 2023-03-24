import pandas as pd
# from django.shortcuts import render_template
from flask import Flask, render_template,request
import pickle
import numpy as np


app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe =pickle.load(open('RegressorModel.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template("index.html", locations= locations)

# @app.route('/predict', methods=['POST','GET'])
# def predict():
#     features = [float(x) for x in request.form.values()]
#     prediction = model.predict([features])[0]
#     return render_template('index.html', prediction=prediction)
@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get("location")
    total_sqft = request.form.get("total_sqft")
    bathrooms=request.form.get("Bathrooms")
    bedrooms=request.form.get("Bedrooms")
    print(location,total_sqft,bathrooms,bedrooms)
    

  
    input = pd.DataFrame([[location,bathrooms,bedrooms,total_sqft]],columns=['location','total_sqft','Bedrooms','Bathrooms'])
    print(input.shape)
    prediction = pipe.predict(input)[0]
    
    


    return str(prediction)
    

if __name__ == '__main__':
    app.run(debug=True, port=5001)
