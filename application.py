from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        driven = request.form.get('kilo_driven')

        # Convert driven to float
        driven = float(driven)

        # Create a DataFrame for prediction
        input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                   data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5))

        # Make the prediction
        prediction = model.predict(input_data)

        # Format the prediction as a string
        prediction_str = "₹ {:.2f}".format(prediction[0])

        return prediction_str

    except Exception as e:
        return str(e), 500  # Return the exception message and set the status code to 500 (Internal Server Error)

if __name__ == '__main__':
    app.run(debug=True)
