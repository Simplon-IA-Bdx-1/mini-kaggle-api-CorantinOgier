#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pandas import read_csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = read_csv('train2.csv', index_col=0)
test = read_csv('test2.csv', index_col=0)

# fillNA
train = train.fillna(0)
test = test.fillna(0)

target_column = 'SeriousDlqin2yrs'

X_train = train.drop(target_column, axis=1).values
y_train = train[target_column].values

X_test = test.drop(target_column, axis=1).values
y_test = test[target_column].values


# Create model from train set

model = RandomForestClassifier() 
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

predictions = pd.DataFrame({'SeriousDlqin2yrs': y_test, 'Predictions': y_pred})    


predictions.to_csv('test2-predictions.csv')

