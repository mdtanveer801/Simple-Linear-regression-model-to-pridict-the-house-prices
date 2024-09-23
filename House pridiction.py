import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn 

dt = pd.read_csv(r"E:\Data Science Notes and projects\PDF and Datasets\House_data.csv")
space=dt['sqft_living']
price=dt['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

pred = regressor.predict(xtest)

plt.scatter(xtrain, ytrain, color= 'pink')
plt.plot(xtrain, regressor.predict(xtrain), color = 'Green')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Visuals for test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()