import pickle
import pandas as pd

example = {"PassengerId": [1040], 'Pclass': [1], 'Name': ['Smith, Mr. Steven'], 'Sex': ['male'], 'Age': [32], 'SibSp': [0], 'Parch': [2], 'Ticket': [''], 'Fare': [10.0], 'Cabin': [''], 'Embarked': ['Q']}

model = pickle.load(open('model.pkl', 'rb'))
pipeline = pickle.load(open('pipeline.pkl', 'rb'))

in_df = pd.DataFrame.from_dict(example)

trans_data = pipeline.transform(in_df)
preds = model.predict(trans_data)
print(preds)