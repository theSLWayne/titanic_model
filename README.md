# Predict Survival on Titanic  

Getting, analyzing, preprocessing the Titanic dataset which can be found [here](https://www.kaggle.com/c/titanic/data).

## How-to get the dataset  

- Install Kaggle Python package

```
pip install kaggle
```

- Download dataset using Kaggle API

```
kaggle competitions download -c titanic
```

## Run the server  

```
python app.py
```

## API  

A RESTful API is used to simulate deployment of a model.  

#### Request

- In order to make a prediction, call a PUT request with a dictionary with all these attributes: ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']  
For an example,  
```json
{
	"PassengerId": 1040, 
	"Pclass": 1, 
	"Name": "Smith, Mr. Steven", 
	"Sex": "male", 
	"Age": 32, 
	"SibSp": 0, 
	"Parch": 2, 
	"Ticket": "", 
	"Fare": 10.0, 
	"Cabin": "", 
	"Embarked": "Q"
}
```

#### Response

- A valid request will result in a dictionary including only one attribute, "prediction" and its value will be the prediction from the values you parsed with the PUT request. If the prediction is 1, the person survived and if it was 0, the person did not survive.  
This is an example response from the model.  
```json
{
    "prediction": 0
}
```

#### Example  

Create a PUT request using __curl__:
```
curl -XPUT -H "Content-type: application/json" -d '{"PassengerId": 1987, "Pclass": 3, "Name": "Sharapova, Ms. Maria", "Sex": "female", "Age": 24, "SibSp": 0, "Parch": 0, "Ticket": "", "Fare": 112.0, "Cabin": "", "Embarked": "S"}' 'http://127.0.0.1:5000/predict'
```  

Then you'll recieve something like the following:  
```
{
    "prediction": 1
}
```

## Files

1. *create_model.py*: Python script that is responisble for transforming the training dataset (using a data transformation pipeline), training the model, saving both pipeline and model as binary files.
2. *app.py*: Python script which contains the Flask server. A basic deployment of the model.
3. *model.pkl*: Binary file containing the trained model.
4. *pipeline.pkl*: Binary file containing data transformation pipeline

## Data Files  

The original dataset comes with 3 files.
1. *train.csv*: The dataset that should be used for training the model.
2. *test.csv*: The dataset taht should be used to evaluate the model. However, It does not contain labels. The test set is only to be used for submitting predictions to the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic).
3. *gender_submission.csv*: An example of what a submission file should look like. Only applicable for the Kaggle competition.

## Models

Here are the models trained to the dataset and their Root Mean Squared Errors

| Model                                  | RMSE               |
|----------------------------------------|--------------------|
| Support Vector Machine Classifier(SVC) | 0.3955054753168236 |
| Random Forest Classifier               | 0.454647017602145  |
| Gaussian Naive Bayes                   | 0.4901259671626783 |
