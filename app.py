import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
prediction = model.predict([[1,0,0.1189,829.10,11.350407,19.48,737,5639.958333,28854,52.1,0,0,0]])
print(prediction)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    for x in request.form.values():
        print(x)

    int_features = [x for x in request.form.values()]

    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    print(output)
    return render_template('index.html', prediction_text='Predicted fully paid should be {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)