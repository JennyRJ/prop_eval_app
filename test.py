import pandas as pd
# from django.shortcuts import render_template
from flask import Flask, render_template

app = Flask(__name__)
data = pd.read_csv(r'C:\Users\JENINE\Desktop\MyApps\Property_Evaluation_WebApp\venv\Cleaned_data.csv')

# df=pickle.load(open('RegressorModel.pkl','rb'))
# # col=['stories_four','stories_one','stories_three','stories_two','lotsize','bedrooms','bathrms','driveway','recroom','fullbase','gashw','airco','garagepl','prefarea']

@app.route('/')
def index():
    neighborhoods = sorted(data['Neighborhood'].unique())
    return render_template("index.html",neighborhoods = neighborhoods)
    print(neighborhoods)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final=[np.array(int_features, dtype=float)]
    prediction=model.predict(final)
    output=round(prediction[0],2)

    return render_template('index.html', pred='The price of your dream house is {} USD Only.'.format(output))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
