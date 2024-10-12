import os
import pickle
import bz2
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load models
pickle_in = bz2.BZ2File(os.path.join('model', 'classification.pkl'), 'rb')
R_pickle_in = bz2.BZ2File(os.path.join('model', 'regression.pkl'), 'rb')
model_C = pickle.load(pickle_in)
model_R = pickle.load(R_pickle_in)

scaler = StandardScaler()  # Ensure your scaler is fitted properly
df = pd.DataFrame({'Temperature': [0], 'Ws': [0], 'FFMC': [0], 'DMC': [0], 'ISI': [0], 'FWI': [0], 'Classes': [0]})
X = df.drop(['FWI', 'Classes'], axis=1)
scaler.fit(X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    final_features = scaler.transform(final_features)
    output = model_C.predict(final_features)[0]
    text = 'Forest is Safe!' if output == 0 else 'Forest is in Danger!'
    return render_template('index.html', prediction_text1=f"{text} --- Chance of Fire is {output}")

@app.route('/predictR', methods=['POST'])
def predictR():
    data = [float(x) for x in request.form.values()]
    data = [np.array(data)]
    data = scaler.transform(data)
    output = model_R.predict(data)[0]
    return render_template('index.html', prediction_text2=f"Fuel Moisture Code index is {output:.4f}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
