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

## Data Files  

The original dataset comes with 3 files.
1. *train.csv*: The dataset that should be used for training the model.
2. *test.csv*: The dataset taht should be used to evaluate the model. However, It does not contain labels. The test set is only to be used for submitting predictions to the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic).
3. *gender_submission.csv*: An example of what a submission file should look like. Only applicable for the Kaggle competition.

## Models

There are two models trained for the Titanic dataset.

1. The model created with Scikit-Learn. It is in the [sklearn_model](sklearn_model)  
  
2. The model created with Tensorflow and Keras. It is in the [tf_model](tf_model)  

Both models are implemented with a simple Flask web app.