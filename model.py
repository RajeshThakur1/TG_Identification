# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('main_df_encoded.csv')

#dataset['experience'].fillna(0, inplace=True)
train,test=train_test_split(dataset,test_size=0.2,random_state=42)


print("Trining Data",train.shape)
print("Testing Dtat",test.shape)
print("Non TG users for Training:-",train[train['is_TG']==0].shape)
print("TG users for Training",train[train['is_TG']==1].shape)
print("Non TG_users for Testing",test[test['is_TG']==0].shape)
print("TG_users for testing",test[test['is_TG']==1].shape)

x_train = train.drop(['is_TG'], axis=1)
y_train = train['is_TG']

x_test = test.drop(['is_TG'], axis=1)
y_test = test['is_TG']
print(x_train.shape)
x_train.drop(['Unnamed: 0'],axis=1,inplace=True)
print(x_train.columns)
print(x_test.shape)
x_test.drop(['Unnamed: 0'],axis=1,inplace=True)
print(x_test.columns)
print(y_train.shape)
print(y_test.shape)


logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(x_train,y_train)
predicted_values=logistic_regression_model.predict(x_test)
true_values = np.array(y_test)
print(true_values)

#dataset['test_score(out of 10)'].fillna(dataset['test_score(out of 10)'].mean(), inplace=True)

print(" Accuracy  {:.2%}".format(accuracy_score(true_values,predicted_values)))

# Saving model to disk

pickle.dump(logistic_regression_model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0,1,2,6,2,2,4,4,3,37,16,47.5]]))
