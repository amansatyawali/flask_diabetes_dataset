from flask import Flask, render_template
from flask.globals import request
import joblib

model = joblib.load('model/diab_79.pkl')

app = Flask(__name__)

@app.route('/')
def home_page() : 
  return render_template('home.html')

@app.route('/userSaved')
def add_user_data() : 
  return render_template('userAdded.html')

@app.route('/input')
def input_patient_data() : 
  return render_template('input.html')

@app.route('/predict', methods = ['post'])
def preditct() : 
  preg = request.form.get('preg')
  plas = request.form.get('plas')
  pres = request.form.get('pres')
  skin = request.form.get('skin')
  test = request.form.get('test')
  mass = request.form.get('mass')
  pedi = request.form.get('pedi')
  age = request.form.get('age')

  prediction = model.predict([[preg, plas, pres, skin, test, mass, pedi, age]])
  if prediction == 1 :
    prediction = 'Diabetic'
  else : 
    prediction = 'Non diabetic'
  return render_template('prediction.html', prediction = prediction)
if __name__ == '__main__' :
  app.run(debug = True)