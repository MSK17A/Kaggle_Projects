"""
This is a python application to solve this competition problem https://www.kaggle.com/competitions/titanic/overview
Titanic - Machine Learning from Disaster
"""
# Import dependencies
import numpy as np
import tensorflow as tf
import pandas as pd  # For playing with CSV files
import helper_functions.utils as utils  # Write any function here
from My_model import *  # To import the model class
from matplotlib import pyplot as plt  # To plot graphs
# I used it to split dataset into Train-Dev-Test sets
from sklearn import model_selection as skMS

# Import the data
df = pd.read_csv("dataset/train.csv")

# Change categorical symbols to numerical-like data
df['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2}, inplace=True)
df['Sex'].replace({'male': 0, 'female': 1}, inplace=True)

# I have taken (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked) as training features
# Drop un used columns
df = df.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket'])
#df = df.dropna()  # I dropped every un registerd data
df = df.fillna(26.3)	# Fill Age NaNs with mean
y = df['Survived'].values  # This is the outputs of the training data
df = df.drop(columns=['Survived'])  # Drop the outputs from the inputs
X = df.values  # Prepared inputs into numpy array


# Splitting dataset into train_set and dev_set
X_train, X_dev, y_train, y_dev = skMS.train_test_split(X, y, test_size=0.2)
X_dev, X_test, y_dev, y_test = skMS.train_test_split(
    X_dev, y_dev, test_size=0.5)

model = My_Model()
print(model.summary())
model.compile(optimizer="adam",
              loss="mse",
              metrics=["mae"])


print("Training the model")


#weights_train = utils.sample_weights(y_train)
#weights_dev = utils.sample_weights(y_dev)

hist = model.fit(
    x=X_train,
    y=y_train,
    batch_size=None,
    epochs=200,
    verbose='auto',
    callbacks=None,
    validation_split=0.0,
    validation_data=(X_dev, y_dev),
    shuffle=True,
    class_weight=None,
    # sample_weight=weights_train,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)

plt.plot(hist.history['loss'])
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['val_mae'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_loss', 'mae', 'val_loss', 'val_mae'], loc='upper left')
plt.show()

print(model.evaluate(X_test, y_test, verbose='auto', return_dict=True))

model.save(
    'Titanic_model',
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
    save_traces=True,
)
