"""
04_Classification.py
"""

'''
문1) bmi.csv 데이터셋을 이용하여 다음과 같이 sigmoid classifier의 모델을 생성하시오. 
   조건1> bmi.csv 데이터셋 
       -> x변수 : 1,2번째 칼럼(height, weight) 
       -> y변수 : 3번째 칼럼(label)
   조건2> 딥러닝 최적화 알고리즘 : Adam
   조건3> learning rage = 0.01    
   조건4> 반복학습 : 2,000번, 200 step 단위로 loss 출력 
   조건5> 최적화 모델 테스트 :  분류정확도(Accuracy report) 출력 
   
 <출력결과>
step = 200 , loss = 0.532565
step = 400 , loss = 0.41763392
step = 600 , loss = 0.34404162
step = 800 , loss = 0.29450226
step = 1000 , loss = 0.25899038
step = 1200 , loss = 0.23218009
step = 1400 , loss = 0.2111086
step = 1600 , loss = 0.19401966
step = 1800 , loss = 0.17981105
step = 2000 , loss = 0.16775638
========================================
accuracy = 0.9601377301019732  
'''
import tensorflow as tf 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import minmax_scale # x변수 정규화 
from sklearn.preprocessing import OneHotEncoder # y data -> one hot
import numpy as np
import pandas as pd
 
# csv file load
bmi = pd.read_csv('C:/ITWILL/6_Tensorflow/data/bmi.csv')
print(bmi.info())

# subset 생성 : label에서 normal, fat 추출 
bmi = bmi[bmi.label.isin(['normal','fat'])] # thin 범주 제외 
print(bmi.head())


# x,y 변수 추출 
X = bmi[['height','weight']] # x변수(1,2칼럼)
y = bmi['label'] # y변수(3칼럼)
X.shape # (15102, 2)
y.shape # (15102,)


# 1. X, y변수 전처리 
x_data = minmax_scale(X) # x_data 정규화 

# y변수 : one-hot encoding 
y_arr = np.array(y) # numpy 변환 
obj = OneHotEncoder()
y_data = obj.fit_transform(y_arr.reshape([-1, 1])).toarray()
y_data.shape # (15102, 2)

# 2. X,Y 변수 정의   
X = tf.constant(x_data, tf.float32) 
y = tf.constant(y_data, tf.float32)
 

# 3. w,b 변수 정의 : 초기값(정규분포 난수 )
w = tf.Variable(tf.random.normal([2, 2]))# [입력수,출력수]
b = tf.Variable(tf.random.normal([2])) # [출력수] 


# 4. 회귀방정식 
def linear_model(X) : # train, test
    y_pred = tf.linalg.matmul(X, w) + b 
    return y_pred # 2차원 


# 5. sigmoid 활성함수 적용 
def sig_fn(X):
    y_pred = linear_model(X)
    sig = tf.nn.sigmoid(y_pred) 
    return sig

# 6. 손실 함수 정의 : 손실계산식 수정 
def loss_fn() : #  인수 없음 
    sig = sig_fn(X) 
    loss = -tf.reduce_mean(y*tf.math.log(sig)+(1-y)*tf.math.log(1-sig))
    return loss

# 7. 최적화 객체 : learning_rate= 0.01
opt = tf.optimizers.Adam(learning_rate=0.01)

# 8. 반복학습 : 반복학습 : 2,000번, 200 step 단위로 loss 출력 
for step in range(2000) :
    opt.minimize(loss=loss_fn, var_list=[w, b])
    
    # 200배수 단위 출력 
    if (step+1) % 200 == 0 :
        print('step =', (step+1), ", loss val = ", loss_fn().numpy())
'''
step = 200 , loss val =  0.44442782
step = 400 , loss val =  0.36517483
step = 600 , loss val =  0.31105947
step = 800 , loss val =  0.27222568
step = 1000 , loss val =  0.24303365
step = 1200 , loss val =  0.22022404
step = 1400 , loss val =  0.20183305
step = 1600 , loss val =  0.18662268
step = 1800 , loss val =  0.17377838
step = 2000 , loss val =  0.16274394
'''

# 9. model 최적화 테스트
y_pred = sig_fn(X) # sigmoid 함수 호출 
print(y_pred)
'''
[[0.19168085 0.7074631 ]
 [0.05956164 0.94630295]
 [0.002904   0.991295  ]
 ...
 [0.25699022 0.65165174]
 [0.65857303 0.429177  ]
 [0.51766473 0.4640499 ]]
'''

# T/F 캐스팅-> 1.0/0.0
y_pred = tf.cast(sig_fn(X).numpy() > 0.5, dtype=tf.float32)
print(y_pred)
'''
[[0. 1.]
 [0. 1.]
 [0. 1.]
 ...
 [0. 1.]
 [1. 0.]
 [1. 0.]], 
'''

acc = accuracy_score(y, y_pred)
print('accuracy =',acc) #accuracy = 0.9682161303138657


###############################################################################


'''
문2) bmi.csv 데이터셋을 이용하여 다음과 같이 softmax classifier 모델을 생성하시오. 
   조건1> bmi.csv 데이터셋 
       -> x변수 : height, weight 칼럼 
       -> y변수 : label(3개 범주) 칼럼
    조건2> w,b 변수 정의    
    조건3> 딥러닝 최적화 알고리즘 : Adam
    조건4> learning rage : 0.001 or 0.005 선택(분류정확도 높은것) 
    조건5> 반복학습, step 단위로 loss : <출력결과> 참고 
    조건6> 분류정확도 출력
    조건7> 앞쪽 예측치와 정답 15개 출력   
    
  <출력 결과>
step = 500 , loss = 0.44498476
step = 1000 , loss = 0.34861678
step = 1500 , loss = 0.28995454
step = 2000 , loss = 0.24887484
step = 2500 , loss = 0.2177721
step = 3000 , loss = 0.19313334
step = 3500 , loss = 0.17303815
step = 4000 , loss = 0.15629826
step = 4500 , loss = 0.1421249
step = 5000 , loss = 0.12996733
========================================
accuracy = 0.9769
========================================
y_pred :  [0 0 1 1 1 1 0 2 0 2 1 2 1 0 2]
y_true :  [0 0 1 1 1 1 0 2 0 2 1 2 1 0 2]  
========================================
'''

import tensorflow as tf # ver1.x
from sklearn.preprocessing import minmax_scale # x data 정규화(0~1)
from sklearn.preprocessing import OneHotEncoder # y data -> one hot
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
 
# dataset load 
bmi = pd.read_csv('C:/ITWILL/6_Tensorflow/data/bmi.csv')
print(bmi.info())

# 칼럼 추출 
col = list(bmi.columns)
print(col) 

# X, y 변수 추출 
X = bmi[col[:2]] # x변수
y = bmi[col[2]] # y변수 

# 1. X, y변수 전처리 

# x_data 정규화 
x_data = minmax_scale(X)


# y변수 : one-hot encoding 
y_arr = np.array(y) # numpy 변환 
obj = OneHotEncoder()
y_data = obj.fit_transform(y_arr.reshape([-1, 1])).toarray()


# 1. X, y변수 전처리 

# x_data 정규화 
x_data = minmax_scale(X)


# y변수 : one-hot encoding 
y_arr = np.array(y) # numpy 변환 
obj = OneHotEncoder()
y_data = obj.fit_transform(y_arr.reshape([-1, 1])).toarray()


# 2. X,Y변수 정의 : 공급형 변수 
X = tf.constant(x_data, tf.float32) 
y = tf.constant(y_data, tf.float32)

# 3. w,b 변수 정의 
tf.random.set_seed(123)
w = tf.Variable(tf.random.normal([2, 3])) # [입력수, 출력수]
b = tf.Variable(tf.zeros([3])) # [출력수]

# 4. 회귀방정식 
def linear_model(X) : # train, test
    y_pred = tf.matmul(X, w) + b  # 행렬곱 : [None,3]*[3,1]=[None,1]
    return y_pred

# 5. softmax 활성함수 적용 
def soft_fn(X):
    y_pred = linear_model(X)
    soft = tf.nn.softmax(y_pred)
    return soft

# 6. 손실 함수 정의 : 손실계산식 수정 
def loss_fn() : #  인수 없음 
    soft = soft_fn(X) # 훈련셋 -> 예측치 : 회귀방정식  
    loss = -tf.reduce_mean(y*tf.math.log(soft)+(1-y)*tf.math.log(1-soft))
    return loss

# 7. 최적화 객체 
optimizer = tf.optimizers.Adam(lr=0.005) 

# 8. 반복학습 
for step in range(5000) : 
    # 오차제곱평균 최적화 : 손실값 최소화 -> [a, b] 갱신(update)
    optimizer.minimize(loss_fn, var_list=[w, b]) #(손실값, 수정 대상)
    
    # 500배수 단위 출력 
    if (step+1) % 500 == 0 :
        print("step =", (step+1), ", loss =", loss_fn().numpy())   


# 9. 최적화된 model 검정 
soft_re = soft_fn(X).numpy()

y_pred = tf.argmax(soft_re, axis=1) # 열축 기준 
y_true = tf.argmax(y, axis=1) #  # 열축 기준 

acc = accuracy_score(y_true, y_pred)
print("="*40)
print('accuracy =', acc) # accuracy = 0.98

# y_true vs y_pred  
print("="*40) 
print('y_pred : ', y_pred.numpy()[:15])
print('y_true : ', y_true.numpy()[:15])

###############################################################################


"""
문3) 다음 digits 데이터셋을 이용하여 다항분류기를 작성하시오.
    <조건1> 아래 <출력결과>를 참고하여 학습율과 반복학습 적용
    <조건2> epoch에 따른 loss value 시각화 
   
 <출력결과>
step = 200 , loss = 0.06003735238669643
step = 400 , loss = 0.02922042555340125
step = 600 , loss = 0.01916724251850193
step = 800 , loss = 0.01418028865527556
step = 1000 , loss = 0.011102086315873883
step = 1200 , loss = 0.008942419709185086
step = 1400 , loss = 0.007311927138572721
step = 1600 , loss = 0.006023632246639046
step = 1800 , loss = 0.004981346240771604
step = 2000 , loss = 0.004163072611802871
========================================
accuracy = 0.9648148148148148
"""

import tensorflow as tf # ver 2.0

from sklearn.datasets import load_digits # dataset 
from sklearn.preprocessing import minmax_scale # x_data -> 0~1
from sklearn.preprocessing import OneHotEncoder # y data -> one hot
from sklearn.metrics import accuracy_score # model 평가 
from sklearn.model_selection import train_test_split # dataset split
import matplotlib.pyplot as plt # loss value 시각화
 
'''
digits 데이터셋 : 숫자 필기체 이미지 -> 숫자 예측(0~9)

•타겟 변수 : y
 - 0 ~ 9 : 10진수 정수 
•특징 변수(64픽셀) : X 
 -0부터 9까지의 숫자를 손으로 쓴 이미지 데이터
 -각 이미지는 0부터 15까지의 16개 명암을 가지는 
  8x8=64픽셀 해상도의 흑백 이미지
'''

# dataset load 
digits = load_digits() # dataset load

X = digits.data  # X변수 
y = digits.target # y변수 
print(X.shape) # (1797, 64) 
print(y.shape) # (1797,)


# 1. X, y변수 전처리  

# X변수 : 정규화
x_data = minmax_scale(X) 

# y변수 : one-hot encoding 
obj = OneHotEncoder()
y_data = obj.fit_transform(y.reshape([-1, 1])).toarray()


# 2. digits dataset split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=123)


# 3. w, b 변수 정의 
w = tf.Variable(tf.random.normal([64, 10], dtype=tf.float64)) # [입력수, 출력수]
b = tf.Variable(tf.random.normal([10], dtype=tf.float64)) # [출력수]


# 5. 회귀방정식 
def linear_model(X) : # train, test
    y_pred = tf.matmul(X, w) + b  
    return y_pred


# 6. softmax 활성함수 적용 
def soft_fn(X):
    y_pred = linear_model(X)
    soft = tf.nn.softmax(y_pred)
    return soft

# 7. 손실 함수 정의 
def loss_fn() : #  인수 없음 
    soft = soft_fn(x_train)   
    loss = -tf.reduce_mean(y_train*tf.math.log(soft)+(1-y_train)*tf.math.log(1-soft))
    return loss


# 8. 최적화 객체 
optimizer = tf.optimizers.Adam(lr=0.01) 


# 9. 반복학습 

    
# 10. 최적화된 model 검증 


# 11. loss value vs epochs 시각화 