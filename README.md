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

## Files  

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
