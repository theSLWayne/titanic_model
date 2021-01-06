#!/usr/bin/env python3

from flask import Flask, request
from flask_restful import Api, Resource
from sklearn.externals import joblib
import pandas as pd

app = Flask(__name__)
api = Api(app)

model = joblib.load('model.pkl')
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
    for i in range(len(prediction)):
      res['predictions'][i + 1] = int(prediction[i])
    return res, 200 # {'prediction': int(prediction[0])}

api.add_resource(Preds, '/predict')

if __name__ == "__main__":
  app.run(debug = True)