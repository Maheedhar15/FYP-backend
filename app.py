from flask import Flask,jsonify
import pickle
import numpy as np
import sklearn

test_data = np.array([[ 1.02446643,  0.06941869,  0.98895712,  0.43198381, -0.20370714,
       -0.08506963, -0.8038884 , -0.2002037 , -0.00346677, -1.71612817,
       -1.38886077, -0.78449469, -0.49678817,  0.11870612]])

filename = './best_model.sav'

loaded_best_model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = loaded_best_model.predict(test_data)
    if(result[0] == 0):
        ans = 'The person is Healthy and is Less prone to Chronic Heart Disease'
    else:
        ans = 'The person is Unhealthy and is more prone to Chronic Heart Disease'
    return jsonify({'prediction': ans})


if  __name__ == "__main__":
    app.run(debug=True)