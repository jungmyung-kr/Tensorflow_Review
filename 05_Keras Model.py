"""
05_Keras Model.py
"""

'''
문) breast_cancer 데이터셋을 이용하여 다음과 같이 keras 모델을 생성하시오.
  조건1> keras layer
       L1 =  30 x 64
       L2 =  64 x 32
       L3 =  32 x 2
  조건2> output layer 활성함수 : sigmoid     
  조건3> optimizer = 'adam',
  조건4> loss = 'binary_crossentropy'
  조건5> metrics = 'accuracy'
  조건6> epochs = 300 
'''

from sklearn.datasets import load_breast_cancer # data set
from sklearn.model_selection import train_test_split # split
from sklearn.preprocessing import minmax_scale 
from tensorflow.keras.utils import to_categorical # one hot encoding
from tensorflow.keras import Sequential # model 생성 
from tensorflow.keras.layers import Dense # DNN layer 

# 1. breast_cancer data load
cancer = load_breast_cancer()

x_data = cancer.data
y_data = cancer.target
print(x_data.shape) # (569, 30) : matrix
print(y_data.shape) # (569,) : vector

# x_data : 정규화 
x_data = minmax_scale(x_data) # 0~1

# y변수 one-hot-encoding 
y_one_hot = to_categorical(y_data)
y_one_hot.shape # (569, 2)


# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_one_hot, test_size = 0.3)


# 3. keras model
model = Sequential() 


# 4. DNN model layer 구축 : hidden layer(2) -> output layer(3)
# hidden layer1 : [30, 64]
model.add(Dense(64, input_shape = (30,), activation = 'relu'))  # 1층 

# hidden layer2 : [64, 32] 
model.add(Dense(32, activation = 'relu')) # 2층 

# output layer : [32, 2]
model.add(Dense(2, activation = 'sigmoid')) # 3층 


# 5. model compile : 학습과정 설정(이항 분류기)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

'''
optimizer : 최적화 알고리즘('adam','sgd') 
loss : 비용 함수('categorical_crossentropy','binary_crossentropy','mse')
metrics : 평가 방법('accuracy', 'mae')
'''

# 6. model training
model.fit(x_train, y_train,
          epochs = 300, # 학습횟수 
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증data
'''
Epoch 100/100
105/105 [==============================] - 0s 893us/sample - loss: 0.1154 - accuracy: 0.9619 
- val_loss: 0.1021 - val_accuracy: 0.9778
#'''

# 7. model evaluation : test dataset
print('optimizered model evaluation')
model.evaluate(x_val, y_val)
# - 0s 67us/sample - loss: 0.1023 - accuracy: 0.9883


###############################################################################


'''
문2) digits 데이터셋을 이용하여 다음과 같이 keras 모델을 생성하시오.
  조건1> keras layer
       L1 =  64 x 32
       L2 =  32 x 16
       L3 =  16 x 10
  조건2> output layer 활성함수 : softmax     
  조건3> optimizer = 'adam',
  조건4> loss = 'categorical_crossentropy'
  조건5> metrics = 'accuracy'
  조건6> epochs = 100 
  조건7> model save : keras_model_digits.h5
'''

import tensorflow as tf # ver 2.0
from sklearn.datasets import load_digits # dataset load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.utils import to_categorical # y변수 one hot
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Dense # model layer
from tensorflow.keras.models import load_model # saved model file -> loading


# 1. digits dataset loading
digits = load_digits()

x_data = digits.data
y_data = digits.target

print(x_data.shape) # (1797, 64) : matrix
print(y_data.shape) # (1797,) : vector

# x_data : 정규화 
x_data = minmax_scale(x_data) # 0~1

# y변수 one-hot-encoding 
y_one_hot = to_categorical(y_data)
y_one_hot
'''
[1., 0., 0., ..., 0., 0., 0.],
'''
y_one_hot.shape # (1797, 10)


# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_one_hot, test_size = 0.3)


# 3. keras model
model = Sequential() 


# 4. DNN model layer 구축 

# hidden layer1 : [64, 32]
model.add(Dense(32, input_shape = (64,), activation = 'relu'))  # 1층 

# hidden layer2 : [32, 16] 
model.add(Dense(16, activation = 'relu')) # 2층 

# output layer : [16, 10]
model.add(Dense(10, activation = 'softmax')) # 3층 


# 5. model compile : 학습과정 설정(다항 분류기)
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])


# 6. model training 
model.fit(x_train, y_train,
          epochs = 100, # 학습횟수 
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증data

# 7. model evaluation : test dataset
print('model evaluation')
model.evaluate(x_val, y_val)

# 8. model save : file save - HDF5 파일 형식 
model.save('keras_model_digits.h5')
print('model saved')



###############################################################################


"""
문3) fashion_mnist 데이터셋을 이용하여 다음과 같이 keras 모델을 생성하시오.
    
  조건1> keras layer
       L1 =  (28, 28) x 128
       L2 =  128 x 64
       L3 =  64 x 32
       L4 =  32 x 16
       L5 =  16 x 10
  조건2> output layer 활성함수 : softmax     
  조건3> optimizer = 'Adam',
  조건4> loss = 'categorical_crossentropy'
  조건5> metrics = 'accuracy'
  조건6> epochs = 15, batch_size = 32   
  조건7> model evaluation : validation dataset
"""
from tensorflow.keras.utils import to_categorical # one hot
from tensorflow.keras.datasets import fashion_mnist # fashion
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Dense, Flatten # model layer
import matplotlib.pyplot as plt

# 1. MNIST dataset loading
(train_img, train_lab),(val_img, val_lab)=fashion_mnist.load_data() # (images, labels)
train_img.shape # (60000, 28, 28) 
train_lab.shape # (60000,) 
 


# 2. x, y변수 전처리 
# x변수 : 정규화(0~1)
train_img = train_img / 255.
val_img = val_img / 255.
train_img[0] # first image(0~1)
val_img[0] # first image(0~1)


# y변수 : one hot encoding 
train_lab = to_categorical(train_lab)
val_lab = to_categorical(val_lab)
val_lab.shape # (10000, 10)

# 입력 : 28x28
# 출력 : 10개 


# 3. keras model
model = Sequential() 
print(model) # object info


# 4. DNN model layer 구축 : hidden layer(3) -> output layer

# [수정] 2차원 image
input_shape = (28, 28) 

# [추가] Flatten layer :2d(28, 28) -> 1d(784)
model.add(Flatten(input_shape = input_shape)) 

# hidden layer1 : [784, 128]  
model.add(Dense(128, input_shape=(784,), activation='relu')) # 1층 
model.add(Dropout(rate = 0.3)) # [추가]

# hidden layer2 : [128, 64]
model.add(Dense(64, activation = 'relu'))  # 2층
model.add(Dropout(rate = 0.1)) # [추가]

# hidden layer3 : [64, 32] 
model.add(Dense(32, activation = 'relu')) # 3층 
model.add(Dropout(rate = 0.1)) # [추가]

# hidden layer4 : [32, 16] 
model.add(Dense(16, activation = 'relu')) # 4층 

# output layer : [16, 10]
model.add(Dense(10, activation = 'softmax')) # 5층 


# 5. model compile : 학습과정 설정(다항 분류기)
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy', # one hot encoding
              metrics = ['accuracy'])


# 6. model training : train(10) -> test(1)
callback = EarlyStopping(monitor='val_loss', patience=5)
# epoch=5 이후 검증 손실이 개선되지 않으면 조기종료 

model_fit = model.fit(train_img, train_lab,
          epochs = 30, # [수정] 학습횟수 : 60000 * 30
          batch_size = 32, # 32 images
          verbose=1,
          validation_data=(val_img, val_lab),# [추가]
          callbacks = [callback]) # [추가] 


# 7. model evaluation : test dataset
print('model evaluation')
model.evaluate(val_img, val_lab)
# - - 1s 133us/sample - loss: 0.2734 - accuracy: 0.8791- 1s 133us/sample - loss: 0.2734 - accuracy: 0.8791


#[내용 추가]
# 1. history : train vs val -> overfitting 시작점 확인 
# 2. dropout : hidden layer에 dropout 적용 
# 3. earlyStopping : 최적의 epochs 찾기 

# 8. model history 
import matplotlib.pyplot as plt 

# loss vs val_loss : overfitting 시작점 : epoch 2
plt.plot(model_fit.history['loss'], 'y', label='train loss')
plt.plot(model_fit.history['val_loss'], 'r', label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend(loc='best')
plt.show()


# accuracy vs val_accuracy : : overfitting 시작점 : epoch 2
plt.plot(model_fit.history['accuracy'], 'y', label='train acc')
plt.plot(model_fit.history['val_accuracy'], 'r', label='val acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show

