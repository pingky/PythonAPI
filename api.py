import pickle
import numpy as np
import pandas as pd
from flask import *
from flask_cors import CORS
import sklearn
from sklearn.linear_model import LinearRegression

model = open('model_regression.pkl','rb')
TheModel = pickle.load(model)
model.close()

app = Flask(__name__)
CORS(app)

@app.route('/single', methods=['POST'])
def classifies():
    fitur = (request.form["Pengalaman"])
    karyawan_baru = [[3]]
    hasil = TheModel.predict(karyawan_baru)
    return hasil.tolist()

@app.route('/bulk', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    try:
        data = pd.read_csv(file)
        dataFitur = data.iloc[:, [2]]
        predictions = TheModel.predict(dataFitur)
        data['Gaji'] = predictions
        # Konversi data gaji ke integer
        data['Gaji'] = data['Gaji'].astype(int)
        data.reset_index(drop=True, inplace=True)        
        print(data)
        return (data.to_json()), 200
    except Exception as e:
        return str(e), 500

@app.route('/json', methods=['POST'])
def classifyJSON():
    try:
        data = request.get_json()
        if not data:
            return "No JSON data provided --- ", 400

        df = pd.DataFrame(data)
        dataFitur = df.iloc[:, [2]]
        predictions = TheModel.predict(dataFitur)
        df['Gaji'] = predictions
        # Convert 'Gaji' to integer
        df['Gaji'] = df['Gaji'].astype(int)
        df.reset_index(drop=True, inplace=True)
        print(df)
        return df.to_json(), 200
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(port=1511)

