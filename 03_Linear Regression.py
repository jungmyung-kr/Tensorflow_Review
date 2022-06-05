"""
03_Linear Regression
"""

'''
문1) 다음과 같이 다중선형회귀방정식으로 모델의 예측치와 오차를 이용하여 
     손실함수를 정의하고 결과를 출력하시오.

    <조건1> w변수[가중치] : Variable()이용 표준정규분포 난수 상수 4개
    <조건2> b변수[편향] : Variable()이용 표준정규분포 난수 상수 1개       
    <조건3> model 예측치 : pred_Y = (X * a) + b -> 행렬곱 함수 적용  
    <조건4> model 손실함수 출력 
        -> 손실함수는 python 함수로 정의 : 함수명 -> loss_fn(err)
    <조건5> 결과 출력 : << 출력 예시 >> 참고     

<< 출력 예시 >>    
w[가중치] =
[[-0.8777014 ]
 [-2.0691    ]
 [-0.47918326]
 [ 1.5532079 ]]
b[편향] = [1.4863125]
Y[정답] = 1.5
pred_Y[예측치] = [[0.7273823]]
loss function = 0.59693813 
'''

import tensorflow as tf 

# 1. X,Y 변수 정의 
X = tf.constant([[1.2, 2.2, 3.5, 4.2]]) # [1,4] - 입력수 : 4개 
Y = tf.constant(1.5) # 출력수(정답) - 1개  

# 2. 가중치, 편향 변수 정의 
w = tf.Variable(tf.random.normal([4, 1])) # 가중치(4,1) 
b = tf.Variable(tf.random.normal([1])) # 편향

# 3. model 예측치/오차/비용
y_pred = tf.matmul(X, w) + b # 예측치
err = tf.subtract(Y, y_pred) # 오차

# 4. 손실함수 정의 
def loss_fn(err) :
    loss = tf.reduce_mean(tf.square(err)) # 비용 : MSE
    return loss

# 5. 결과 출력 
print('w[가중치] =')
print(w.numpy())
print('b[편향] =', b.numpy())

print('Y[정답] =', Y.numpy())
print('pred_Y[예측치] =', y_pred.numpy())
print('errpr =', err.numpy())
print('loss function =', loss_fn(err).numpy())


###############################################################################


'''
문2) women.csv 데이터 파일을 이용하여 선형회귀모델 생성하시오.
     <조건1> x변수 : height,  y변수 : weight
     <조건2> learning_rate=0.5
     <조건3> 최적화함수 : Adam
     <조건4> 반복학습 : 200회
     <조건5> 학습과정 출력 : step, loss_value
     <조건6> 최적화 모델 검증 : MSE, 회귀선 시각화  
'''
import tensorflow as tf # ver2.0 

import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

women = pd.read_csv('C:/ITWILL/5_Tensorflow/data/women.csv')
print(women.info())
print(women)

# 1. x,y data 생성 
x_data = women['height']
y_data = women['weight']

# 정규화 
print(x_data.max()) # 72
print(y_data.max()) # 164

# 2. 정규화(0~1)
X = x_data / 72
Y = y_data / 164
X.dtype # dtype('float64')
Y.dtype # dtype('float64')

X = tf.constant(X, dtype=tf.float32)
Y = tf.constant(Y, dtype=tf.float32)

# 3. w,b변수 정의 - 난수 이용 
w = tf.Variable(tf.random.uniform([1], 0.1, 1.0))
b = tf.Variable(tf.random.uniform([1], 0.1, 1.0))
print(w) # float32

# 4. 회귀모델 
def linear_model(X) : # 입력 X
    y_pred = tf.multiply(X, w) + b # y_pred = X * a + b
    return y_pred

# 5. 비용 함수 정의 : 예측치 > 오차 > 손실함수 
def loss_fn() : #  인수 없음 
    y_pred = linear_model(X) # 예측치 : 회귀방정식  
    err = Y - y_pred # 오차 
    loss = tf.reduce_mean(tf.square(err)) # 오차제곱평균(MSE) 
    return loss

# 6. model 최적화 객체 : 오차의 최소점을 찾는 객체  


# 7. 반복학습 : 200회 


# 8. 최적화된 model 검증

# 1) MSE 평가 


# 2) 회귀선    


###############################################################################


'''
문3) load_boston 데이터셋을 이용하여 다음과 같이 선형회귀모델 생성하시오.
     <조건1> x변수 : boston.data,  y변수 : boston.target
     <조건2> w변수, b변수 정의 : tf.random.normal() 이용 
     <조건3> learning_rate=0.5
     <조건4> 최적화함수 : Adam
     <조건5> 학습 횟수 1,000회
     <조건6> 학습과정과 MSE 출력 : <출력결과> 참고 
     
 <출력결과>
step = 100 , loss = 4.646641273041386
step = 200 , loss = 1.1614418341428459
step = 300 , loss = 0.40125618834653615
step = 400 , loss = 0.21101471610402903
step = 500 , loss = 0.13666187210671069
step = 600 , loss = 0.09779346604325287
step = 700 , loss = 0.07608768653282329
step = 800 , loss = 0.06372023833861612
step = 900 , loss = 0.0566559217407318
step = 1000 , loss = 0.05266675679250506
=============================================
MSE= 0.04122129293175945
'''
import tensorflow as tf # ver2.0
# pip install sklearn
from sklearn.model_selection import train_test_split # datast splits
from sklearn.metrics import mean_squared_error # model 평가 
from sklearn.datasets import load_boston
from sklearn.preprocessing import minmax_scale # 정규화(0~1) 

# 1. data loading
boston = load_boston()

# 변수 선택 
X = boston.data # x 
y = boston.target # y : 숫자 class(0~2)

print(X.shape) # (506, 13)
print(y.shape) # (506,)

# y변수 정규화 
y = minmax_scale(y)

# 2. train/test split(70 vs 30)
x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123)

# 3. w, b변수 정의 : tf.random.normal() 함수 이용 

# 4. 회귀모델 : 행렬곱 

# 5. 비용 함수 정의 : 예측치 > 오차 > 손실함수 

# 6. 최적화 객체 

# 7. 반복학습 

# 8. 최적화된 model 평가


###############################################################################


"""
문4) boston 데이터셋을 이용하여 다음과 같이 Keras DNN model layer을 
    구축하고, model을 학습하고, 검증(evaluation)하시오. 
    <조건1> 4. DNN model layer 구축 
         1층(hidden layer1) : units = 64
         2층(hidden layer2) : units = 32
         3층(hidden layer3) : units = 16 
         4층(output layer) : units=1
    <조건2> 6. model training  : 훈련용 데이터셋 이용 
            epochs = 50
    <조건3> 7. model evaluation : 검증용 데이터셋 이용     
"""
from sklearn.datasets import load_boston  # dataset
from sklearn.model_selection import train_test_split # split
from sklearn.preprocessing import minmax_scale # 정규화(0~1) 
from sklearn.metrics import mean_squared_error, r2_score


# keras model 관련 API
import tensorflow as tf # ver 2.0
from tensorflow.keras import Sequential # model 생성 
from tensorflow.keras.layers import Dense # DNN layer
print(tf.keras.__version__) # 2.2.4-tf

# 1. x,y data 생성 
X, y = boston = load_boston(return_X_y=True)
X.shape # (442, 10)
y.shape # (442,)

# y 정규화 
X = minmax_scale(X)
y = minmax_scale(y)

# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size = 0.3, random_state=1)
x_train.shape 
y_train.shape 


# 3. keras model
model = Sequential() 
print(model) # object info


# 4. DNN model layer 구축 


# 5. model compile : 학습과정 설정(다항 분류기)



# 6. model training 



# 7. model evaluation : test dataset



    




