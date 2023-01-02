import pandas as pd
# from django.shortcuts import render_template
from flask import Flask, render_template,request
import pickle
import numpy as np


app = Flask(__name__)
data = pd.read_csv(r'C:\Users\JENINE\Desktop\MyApps\Property_Evaluation_WebApp\venv\Cleaned_data.csv')
pipe =pickle.load(open(r'C:\Users\JENINE\Desktop\MyApps\Property_Evaluation_WebApp\venv\RegressorModel.pkl', 'rb'))

@app.route('/')
def index():
    neighborhoods = sorted(data['Neighborhood'].unique())
    return render_template("index.html", neighborhoods= neighborhoods)
    


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    neighborhood = request.form.get("Neighborhood")
    sqft = request.form.get("sq_mtrs")
    bathrooms=request.form.get("Bathrooms")
    bedrooms=request.form.get("Bedrooms")
    
    input = pd.DataFrame([[Neighborhood,Bathrooms,Bedrooms,sq_mtrs]],columns=['Neighborhood','Bathrooms','Bedrooms','sq_mtrs'])
    prediction = pipe.predict(input)[0]


    return render_template('index.html', pred='The price of your dream house is {} USD Only.'.format(output))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
