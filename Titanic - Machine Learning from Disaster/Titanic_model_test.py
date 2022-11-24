import numpy as np
import tensorflow as tf
import pandas as pd
import helper_functions.utils as utils

model = tf.keras.models.load_model('Titanic_model')
model.summary()

df = pd.read_csv("dataset/test.csv")

df['Embarked'].replace({'S':0, 'C':1, 'Q':2}, inplace=True)
df['Sex'].replace({'male':0, 'female':1}, inplace=True)
df = df.fillna(26.3)

# I have taken (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
Xdf = df.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket'])
X_test = Xdf.values

Preds = model.predict(X_test)
Predictions = []

for pred in range(0, len(Preds)):
    if (Preds[pred] >= 0.5):
        Predictions.append(1)
    else:
        Predictions.append(0)

data = {'PassengerId': df['PassengerId'].values, 'Survived': Predictions}
df_submission = pd.DataFrame(data=data)
df_submission.to_csv(path_or_buf='submission.csv', index=False)