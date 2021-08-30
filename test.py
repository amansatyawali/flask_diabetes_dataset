import joblib

dib_model = joblib.load('diab_79.pkl')

prediction = dib_model.predict([[1,85,66,29,0,26.6,0.351,31]])

if prediction[0] == 0 :
  print('Non diabetic')
else :
  print('Diabetic')