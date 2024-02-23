from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Sequential import *
from Dense import *
import numpy as np
import pandas as pd
import matplotlib as plt
from evalvalid import *
from timeit import default_timer as timer
# import tensorflow as tf

# def random_data_generator():
#     """
#     (x1 + x2)^2
#     """
#     x1 = (np.random.rand(320)*10)
#     x2 = (np.random.rand(320)*10)
#     return [x1.T,x2.T,((x1 + x2)**2).T]

# sc = MinMaxScaler()


# x1, x2, y = random_data_generator()
# data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
# # print(data)
# X = data.drop(columns=['y'])
# y = data['y']
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)



# print(X_train.shape)

def main():
    data = pd.read_csv("train_scaled.csv")

    # print(data)

    X = data.drop(columns=['Exited','CustomerId','IsSenior','IsActive_by_CreditCard','Products_Per_Tenure','female'])
    y = data['Exited']

    print(data.columns)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

    X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.2)

    # print(type(X_train), type(X_test), type(y_train), type(y_test))

    print(X_train.shape, X_test.shape)

    model = Sequential()

    # model.add(Dense(20,"relu"))
    # model.add(Dense(18,"relu"))
    # model.add(Dense(15,"relu"))
    # model.add(Dense(12,"relu"))
    # model.add(Dense(1024,"lrelu"))
    model.add(Dense(512,"lrelu"))
    model.add(Dense(256,"lrelu"))
    model.add(Dense(128,"relu"))
    model.add(Dense(1,"sigmoid"))
    model.summary()
    start = timer()
    model.train(X_train, y_train, epochs = 50, learning_rate = 20, val_set=(X_test,y_test),
                batch_size = 512, loss = "BinaryCrossEntropy", verbose = 1)
    y_predicted = model.predict(X_test)
    print("zaman: ", (timer() - start))
    # for i,j in zip(y_predicted, y_test):
    #     print(i, j)
    eval_valid(y_predicted, y_test)
    # print(calculate_accuracy(y_predicted, y_test))
    plt.plot(model.errors,label="err")
    plt.plot(model.val_errors,label="val_err")
    plt.legend()
    plt.show()

# main()


sc_X = StandardScaler()

data = pd.read_csv("islemciveri.csv")
data['Date'] = pd.to_datetime(data['Date'], format='%A, %B %d, %Y')
data['Date'] = data['Date'].apply(lambda x: x.toordinal())
X = data[['Date']][:365]
y = data['Price_scaled'][:365]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

tahmin = data[['Date']][365:]

X_train_sc = sc_X.fit_transform(X_train)
# y_train = sc_Y.fit_transform(y_train)

X_test_sc = sc_X.transform(X_test)
# y_test = sc_Y.transform(y_test)

tahmin_sc = sc_X.transform(tahmin)

model = Sequential()
model.add(Dense(128,"relu"))
model.add(Dense(64,"relu"))
model.add(Dense(32,"relu"))
model.add(Dense(16,"relu"))
model.add(Dense(1,"none"))
model.train(X_train_sc, y_train, epochs= 100000,
            learning_rate = 0.0008,
            batch_size = 32,
            verbose=25,
            loss="MSE")

y_train_pred = model.predict(X_train_sc) * 10000
y_pred = model.predict(X_test_sc) * 10000

y_pred_next = model.predict(tahmin_sc) * 10000

plt.plot(X['Date'], (y * 10000), label='True Values', color ='blue')

plt.scatter(X_test['Date'], y_pred, color='red',label='Predicted Test')
plt.scatter(X_train['Date'], y_train_pred, color='yellow',label='Predicted Train')
plt.scatter(tahmin['Date'], y_pred_next, color="black", label='Nisana kadar')

plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.close()
# plt.plot(model.errors[100:])
plt.show()