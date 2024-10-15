from flask import Flask, request, render_template
from joblib import load
import numpy as np


app = Flask(__name__)


model = load('cropmodel.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        # inputs = [
        #     float(request.form['nitrogen']),
        #     float(request.form['phosphorus']),
        #     float(request.form['potassium']),
        #     float(request.form['temperature']),
        #     float(request.form['humidity']),
        #     float(request.form['ph']),
        #     float(request.form['rainfall'])
        # ]

        
        inputs=[float(x) for x in request.form.values() ]       
         
        input_array = np.array([inputs])

        
        prediction = model.predict(input_array)[0]

        
        return render_template('index.html', prediction_text=f'Recommended Crop: {prediction}')
    except Exception as e:
        return f'Error: {e}'

if __name__ == '__main__':
    app.run(debug=True)
