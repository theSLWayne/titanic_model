#!/usr/bin/env python3

from flask import Flask, request
from flask_restful import Api, Resource
from sklearn.externals import joblib
import pandas as pd
from tensorflow import keras

app = Flask(__name__)
api = Api(app)

model = keras.models.load_model('titanic_model')
pipeline = joblib.load('pipeline.pkl')

def islist(obj):
  return True if ("list" in str(type(obj))) else False

class Preds(Resource):
  def put(self):
    json_ = request.json
    if islist(json_['PassengerId']):
      entry = pd.DataFrame(json_)
    else:
      entry = pd.DataFrame([json_])
    entry_transformed = pipeline.transform(entry)
    prediction = model.predict(entry_transformed)
    res = {'predictions': {}}
    print(prediction)
    for i in range(len(prediction)):
      res['predictions'][i + 1] = 1 if prediction[i] >= 0.5 else 0
    return res, 200

api.add_resource(Preds, '/predict')

if __name__ == "__main__":
  app.run(debug = True)