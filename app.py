import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
#run this app.py through command prompt (python app.py)
app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('D:\datasets\SPAM_HAM\spam_detector.pickle', 'rb'))
cvm=pickle.load(open('D:\datasets\SPAM_HAM\cv.pickle', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features = cvm.transform(features)
    prediction = model.predict(final_features)[0]
    output = prediction
    if output==0:
        output='HAM'
    else:
        output='SPAM'

    return render_template('index.html', prediction_text='The message is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
    
    
