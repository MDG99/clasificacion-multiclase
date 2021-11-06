def ova_gradient_descent(X, Y, eta, epochs, percent):
    '''Esta funcion se utiliza para implimentar el método de Clasificación One vs All con Gradiente Descent
    ova_gradient_descent(X, Y, eta, epocs) where:
    X: DataFrame de instancias o features
    Y: DataFrame de targets
    eta: tasa de aprendizaje (learning rate)
    epochs: numero máximo de iteraciones
    percent: % de datos que seran utilizados para el test (base 100)
    
    ------------------------------------
    Return:
    In order: theta, test_index, train_index, Y_predict, J_log
    
    theta: valores correspondientes a theta_n
    test_index: data test index
    train_index: data training index
    Y_predict: Y predict values
    J_log: errores por numero de epoca
    Prob: Probabilidades de los 3 modelos   
    '''
    import numpy as np
    import pandas as pd
    import random
    
    m = len(X)
    test_index = list(pd.Series(random.sample(list(np.arange(0, m)), round(m * percent / 100))).sort_values())
    train_index = list(np.arange(0, m))
    
    for element in test_index:
        train_index.remove(element)
        
    
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = np.c_[Y.iloc[train_index]]
    Y_test = np.c_[Y.iloc[test_index]]
    
    # Entrenamiento
    theta = np.array([np.zeros(5).reshape(-1,1), np.zeros(5).reshape(-1,1), np.zeros(5).reshape(-1,1)])
    yc = np.array([np.zeros(len(Y_train)).reshape(-1,1), np.zeros(len(Y_train)).reshape(-1,1), np.zeros(len(Y_train)).reshape(-1,1)])
   
    for i in range(3):
        yc[i] = (Y_train == i).astype(np.int32).reshape(-1,1)
        theta[i] = np.random.randn((X.shape[1] + 1), 1)    

    m = len(X_train)
    
    X_b = np.c_[np.ones((m, 1)), X_train]
        
    for i in range(3):
        J_log = np.zeros(epochs)
        theta_t = theta[i]
        y_train_t = yc[i]
        for j in range(epochs):
            J_log[j] =(-1/m)*(y_train_t*np.log(sigmoid(X_b @ theta_t)) + (1-y_train_t)*(np.log(1-sigmoid(X_b @ theta_t)))).sum(axis=0)
            gradients = (1 / m) * (X_b.T @ (sigmoid(X_b @ theta_t) - y_train_t))
            theta_t = theta_t - eta * gradients                       
        theta[i] = theta_t                               

    # Test
    
    m = len(X_test)
    
    X_b_test = np.c_[np.ones((m, 1)), X_test]
    
    y_pr0 = np.zeros(len(test_index))
    y_pr1 = np.zeros(len(test_index))
    y_pr2 = np.zeros(len(test_index))
   
    y_pr= np.array([y_pr0, y_pr1, y_pr2])  
    
    for i in range(3):
        y_pr[i] = sigmoid(theta[i].T @ X_b_test.T)
        
    Prob = np.c_[y_pr[0:1].T , y_pr[1:2].T , y_pr[2:3].T ]    
    Y_predict = np.argmax(Prob, axis=-1)    

    return theta, test_index, train_index, Y_predict, J_log, Prob, Y_test

def sigmoid(z):
    import numpy as np
    return 1/(1+np.exp(-z))

def ovo_gradient_descent(X, Y, eta, epochs, percent):
    '''Esta funcion se utiliza para implimentar el método de Clasificación One Vs One con Gradiente Descent
    ovo_gradient_descent(X, Y, eta, epocs) where:
    X: DataFrame de instancias o features
    Y: DataFrame de targets
    eta: tasa de aprendizaje (learning rate)
    epochs: numero máximo de iteraciones
    percent: % de datos que seran utilizados para el test (base 100)
    
    ------------------------------------
    Return:
    In order: theta, test_index, train_index, Y_predict, J_log
    
    theta: valores correspondientes a theta_n
    test_index: data test index
    train_index: data training index
    Y_predict: Y predict values
    J_log: errores por numero de epoca
    Prob: Probabilidades de los 3 modelos
    '''
    import numpy as np
    import pandas as pd
    import random
    
    m = len(X)
    test_index = list(pd.Series(random.sample(list(np.arange(0, m)), round(m * percent / 100))).sort_values())
    train_index = list(np.arange(0, m))
    
    for element in test_index:
        train_index.remove(element)
        
    
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = np.c_[Y.iloc[train_index]]
    Y_test = np.c_[Y.iloc[test_index]]
    
    A = pd.concat([X.iloc[train_index], Y.iloc[train_index]] , axis = 1)
    B = pd.concat([X.iloc[test_index], Y.iloc[test_index]] , axis = 1)
    
    Xtrain_zero = A[A.iloc[:,4] == 0 ].iloc[:,[0,1,2,3]]      # X_train de Clase 0 
    Xtrain_one = A[A.iloc[:,4] == 1 ].iloc[:,[0,1,2,3]]       # X_train de Clase 1 
    Xtrain_two = A[A.iloc[:,4] == 2 ].iloc[:,[0,1,2,3]]       # X_train de Clase 2
    Ytrain_zero = A[A.iloc[:,4] == 0 ].iloc[:,4]      # Y_train de Clase 0 
    Ytrain_one = A[A.iloc[:,4] == 1 ].iloc[:,4]       # Y_train de Clase 1 
    Ytrain_two = A[A.iloc[:,4] == 2 ].iloc[:,4]       # Y_train de Clase 2
    
    Ytest_zero = B[B.iloc[:,4] == 0 ].iloc[:,4]      # Y_test de Clase 0 
    Ytest_one = B[B.iloc[:,4] == 1 ].iloc[:,4]       # Y_test de Clase 1 
    Ytest_two = B[B.iloc[:,4] == 2 ].iloc[:,4]       # Y_test de Clase 2
    
    Xtrain01 = np.append(Xtrain_zero, Xtrain_one, axis=0)
    Xtrain02 = np.append(Xtrain_zero, Xtrain_two, axis=0)
    Xtrain12 = np.append(Xtrain_one, Xtrain_two, axis=0)
    Ytrain01 = np.append(Ytrain_zero, Ytrain_one, axis=0)
    Ytrain02 = np.append(Ytrain_zero, Ytrain_two, axis=0)
    Ytrain12 = np.append(Ytrain_one, Ytrain_two, axis=0)
    
    Ytest01 = np.append(Ytest_zero, Ytest_one, axis=0)
    Ytest02 = np.append(Ytest_zero, Ytest_two, axis=0)
    Ytest12 = np.append(Ytest_one, Ytest_two, axis=0)
    
    # Entrenamiento
    theta = np.array([np.zeros(5).reshape(-1,1), np.zeros(5).reshape(-1,1), np.zeros(5).reshape(-1,1)])
    for i in range(3):
        theta[i] = np.random.randn((X.shape[1] + 1), 1) 
    
    yc01 = (Ytrain01 == 0).astype(np.int32).reshape(-1,1)
    yc02 = (Ytrain02 == 0).astype(np.int32).reshape(-1,1)
    yc12 = (Ytrain12 == 1).astype(np.int32).reshape(-1,1)

    ytest01 = (Ytest01 == 0).astype(np.int32).reshape(-1,1)
    ytest02 = (Ytest02 == 0).astype(np.int32).reshape(-1,1)
    ytest12 = (Ytest12 == 1).astype(np.int32).reshape(-1,1)
    
    #b = (Y_train == 0)
    
    Xtrain_zero = A[A.iloc[:,4] == 0 ].iloc[:,[0,1,2,3]]      # X_train de Clase 0 
    Xtrain_one = A[A.iloc[:,4] == 1 ].iloc[:,[0,1,2,3]]       # X_train de Clase 1 
    Xtrain_two = A[A.iloc[:,4] == 2 ].iloc[:,[0,1,2,3]]       # X_train de Clase 2
    
    Xtrain01 = np.append(Xtrain_zero, Xtrain_one, axis=0)
    Xtrain02 = np.append(Xtrain_zero, Xtrain_two, axis=0)
    Xtrain12 = np.append(Xtrain_one, Xtrain_two, axis=0)
    
    yc = np.array([yc01,yc02,yc12], dtype=object)
    Y_c= np.array([ytest01,ytest02,ytest12], dtype=object)
    
    #Xtrain = np.array([Xtrain01, Xtrain01, Xtrain12])
    
    m = len(X_train)
    m01 = len(Xtrain01)
    m02 = len(Xtrain02)
    m12 = len(Xtrain12)
    
    X_b01 = np.c_[np.ones((m01, 1)), Xtrain01]              #Clase 0 vs 1
    X_b02 = np.c_[np.ones((m02, 1)), Xtrain02]              #Clase 0 vs 2
    X_b03 = np.c_[np.ones((m12, 1)), Xtrain12]              #Clase 1 vs 2
    
    
    theta1 = theta[0]
    y_train1 = yc[0]
    J_log1 = np.zeros(epochs)
    for i in range(epochs):
        J_log1[i] =(-1/m)*(y_train1*np.log(sigmoid(X_b01 @ theta1)) + (1-y_train1)*(np.log(1-sigmoid(X_b01 @ theta1)))).sum(axis=0)
        gradients = (1 / m) * (X_b01.T @ (sigmoid(X_b01 @ theta1) - y_train1))
        theta1 = theta1- eta * gradients                       
    theta[0] = theta1
             
    theta2 = theta[1]
    y_train2 = yc[1]
    J_log2 = np.zeros(epochs)
    for j in range(epochs):
        J_log2[i] =(-1/m)*(y_train2*np.log(sigmoid(X_b02 @ theta2)) + (1-y_train2)*(np.log(1-sigmoid(X_b02 @ theta2)))).sum(axis=0)
        gradients = (1 / m) * (X_b02.T @ (sigmoid(X_b02 @ theta2) - y_train2))
        theta2 = theta2- eta * gradients                       
    theta[1] = theta2
    
    theta3 = theta[2]
    y_train3 = yc[2]
    J_log3 = np.zeros(epochs)
    for j in range(epochs):
        J_log3[i] =(-1/m)*(y_train3*np.log(sigmoid(X_b03 @ theta3)) + (1-y_train3)*(np.log(1-sigmoid(X_b03 @ theta3)))).sum(axis=0)
        gradients = (1 / m) * (X_b03.T @ (sigmoid(X_b03 @ theta3) - y_train3))
        theta3 = theta3- eta * gradients                       
    theta[2] = theta3
    
    J_log = np.c_[J_log1 , J_log2 , J_log3]  
    # Test
    
    m = len(X_test)
    
    X_b_test = np.c_[np.ones((m, 1)), X_test]
    
    y_pr0 = np.zeros(len(test_index))
    y_pr1 = np.zeros(len(test_index))
    y_pr2 = np.zeros(len(test_index))
   
    y_pr= np.array([y_pr0, y_pr1, y_pr2])  
    
    for i in range(3):
        y_pr[i] = sigmoid(theta[i].T @ X_b_test.T)
        
    Prob = np.c_[y_pr[0:1].T , y_pr[1:2].T , y_pr[2:3].T ]  
    y_01_predict = y_pr[0:1].T.round(0)
    y_02_predict = y_pr[1:2].T.round(0)
    y_12_predict = y_pr[2:3].T.round(0)
    
    Y_predict=np.zeros(len(test_index))
    for i in range(len(test_index)):
        if (y_01_predict[i] == 1 and y_02_predict[i] == 1):
            Y_predict[i] = 0
        if (y_01_predict[i] == 0 and y_12_predict[i] == 1):
            Y_predict[i] = 1
        if (y_02_predict[i] == 0 and y_12_predict[i] == 0):
            Y_predict[i] = 2
    #Y_predict = np.argmax(Prob, axis=-1)    

    return theta, test_index, train_index, Y_predict, J_log, Prob, Y_test, Y_c






