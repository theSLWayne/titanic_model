#!/usr/bin/env python3

from flask import Flask, request
from flask_restful import Api, Resource
from sklearn.externals import joblib
import pandas as pd

app = Flask(__name__)
api = Api(app)

model = joblib.load('model.pkl')
pipeline = joblib.load('pipeline.pkl')

class Preds(Resource):
  def put(self):
    json_ = request.json
    entry = pd.DataFrame([json_])
    entry_transformed = pipeline.transform(entry)
    prediction = model.predict(entry_transformed)
    return {'prediction': int(prediction[0])}, 200

api.add_resource(Preds, '/predict')

if __name__ == "__main__":
  app.run(debug = True)