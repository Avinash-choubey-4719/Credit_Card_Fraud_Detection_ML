# -*- coding: utf-8 -*-
"""
Created on Sun May  8 13:46:48 2022

@author: DELL
"""
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 



credit_card_dataset = pd.read_csv("creditcard.csv")


legit = credit_card_dataset[credit_card_dataset.Class == 0]
fraud = credit_card_dataset[credit_card_dataset.Class == 1]



legit_sample = legit.sample(n = 492)



new_dataset = pd.concat([legit_sample, fraud], axis = 0)



x = new_dataset.drop(columns = 'Class', axis = 1)
y = new_dataset['Class']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 2)


model = LogisticRegression()

model.fit(x_train, y_train)



x_train_prediction = model.predict(x_train)
x_train_accuracy = accuracy_score(x_train_prediction, y_train)



x_test_prediction = model.predict(x_test)
x_test_accuracy = accuracy_score(x_test_prediction, y_test)


