"""
06_CNN Model.py
"""

'''
문1) 다음과 같이 합성곱층과 폴링층을 작성하고 결과를 확인하시오.
  <조건1> input image : mnist.train의 7번째 image     
  <조건2> input image shape : [-1, 28,28,1] 
  <조건3> 합성곱 
         - strides= 2x2, padding='SAME'
         - Filter : 3x3, 특징맵 = 10
  <조건4> Max Pooling 
    -> ksize= 3x3, strides= 2x2, padding='SAME' 
'''

import tensorflow as tf 

from tensorflow.keras.datasets.mnist import load_data # ver2.0 dataset
import numpy as np
import matplotlib.pyplot as plt

# 1. image read  
(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape) # (60000, 28, 28)

# 2. 실수형 변환 : int -> float
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')

# 3. 정규화 
x_train = x_train / 255 # x_train = x_train / 255
x_test = x_test / 255

# 7번째 image 
img = x_train[6]
plt.imshow(img, cmap='gray') # 숫자 5 -> x-ray 방식 
plt.show()

# input image
img = img.reshape(-1, 28,28,1) 

# Filter
Filter = tf.Variable(tf.random.normal([3,3,1,10], stddev=0.01))

# 합성곱  
conv2d = tf.nn.conv2d(img, Filter, strides=[1,2,2,1], padding='SAME')
print(conv2d)
# Tensor("Conv2D:0", shape=(1, 14, 14, 10), dtype=float32)

conv2d_img = np.swapaxes(conv2d, 0, 3)
for i, one_img in enumerate(conv2d_img) : 
    # 1행5열 : 5개 이미지 
    plt.subplot(1,10,i+1)
    plt.imshow(one_img.reshape(14,14), cmap='gray')    
plt.show()    

# Max Pooling
pool = tf.nn.max_pool(conv2d, ksize=[1,3,3,1], strides=[1,2,2,1], 
                      padding='SAME')
print(pool)
#Tensor("MaxPool:0", shape=(1, 7, 7, 10), dtype=float32)

pool_img = np.swapaxes(pool, 0, 3)

for i, one_img in enumerate(pool_img) :
    # 1행5열 : 5개 이미지 
    plt.subplot(1,10, i+1)
    plt.imshow(one_img.reshape(7,7), cmap='gray')
plt.show()


###############################################################################


'''
문2) 다음과 같이 Convolution layer와 Max Pooling layer를 정의하고, 실행하시오.
  <조건1> input image : volcano.jpg 파일 대상    
  <조건2> Convolution layer 정의 
    -> Filter : 6x6
    -> featuremap : 16개
    -> strides= 1x1, padding='SAME'  
  <조건3> Max Pooling layer 정의 
    -> ksize= 3x3, strides= 2x2, padding='SAME' 
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('C:/ITWILL/6_Tensorflow/data/images/volcano.jpg') # 이미지 읽어오기
plt.imshow(img)
plt.show()
print(img.shape)  # (405, 720, 3) (512, 768, 3) - 1차원

# input image(X) : [size, h, w, color] 
Img = img.reshape(-1, 405, 720, 3) # 전체,세로,가로,색상 
print(type(Img))
print(Img)

# TypeError 해결방법 
Img = Img.astype('float32')
print(Img)
'''
TypeError: Value passed to parameter 'input' has DataType uint8 not in list of allowed values: float16, bfloat16, float32, float64
'''

# Filter(weight) 변수 정의 : [h, w, color, featuremap]
Filter = tf.Variable(tf.random.normal([6, 6, 3, 16]))

# Convolution layer 
conv2d = tf.nn.conv2d(Img, Filter, strides=[1,1,1,1], padding='SAME')
print(conv2d)
# Tensor("Conv2D_10:0", shape=(1, 405, 720, 16), dtype=float32)

# Max Pooling layer
pool = tf.nn.max_pool(conv2d, ksize=[1,3,3,1], strides=[1,2,2,1], 
                      padding='SAME')
print(pool)
#Tensor("MaxPool_2:0", shape=(1, 203, 360, 16), dtype=float32)

    
# 합성곱 연산 
conv2d_img = np.swapaxes(conv2d, 0, 3)

for i, one_img in enumerate(conv2d_img) : 
    plt.subplot(1,16,i+1)
    plt.imshow(one_img.reshape(405, 720))    
plt.show()    

# 폴링 연산
pool_img = np.swapaxes(pool, 0, 3)

for i, one_img in enumerate(pool_img) :
    plt.subplot(1,16, i+1)
    plt.imshow(one_img.reshape(203, 360))
plt.show()


###############################################################################


'''
문3) 다음과 같은 조건으로 keras CNN model layer를 작성하시오.

1. Convolution1
    1) 합성곱층 
      - filters=64, kernel_size=5x5, padding='same'  
    2) 풀링층(max) 
     - pool_size= 2x2, strides= 2x2, padding='same'

2. Convolution2
    1) 합성곱층 
      - filters=128, kernel_size=5x5, padding='same'
    2) 풀링층
     - pool_size= 2x2, strides= 2x2, padding='same'
    
3. Flatten layer 

4. Affine layer(Fully connected)
    - 256 node, activation = 'relu'
    - Dropout : 0.5%

5. Output layer(Fully connected)
    - 10 node, activation = 'softmax'
    
--------------------------------------------------------
6. model training 
   - epochs=3, batch_size=100   

7. model evaluation
   - model.evaluate(x=x_val, y=y_val)                   
'''

from tensorflow.keras.datasets.mnist import load_data # ver2.0 dataset
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation # Convolution layer
from tensorflow.keras.layers import Dense, Dropout, Flatten # Affine layer

# minst data read
(x_train, y_train), (x_val, y_val) = load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,) : 10진수 

# image data reshape : [s, h, w, c]
x_train = x_train.reshape(60000, 28, 28, 1)
x_val = x_val.reshape(10000, 28, 28, 1)
print(x_train.shape) # (60000, 28, 28, 1)
print(x_train[0]) # 0 ~ 255

# x_data : int -> float
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')


# x_data : 정규화 
x_train /= 255 # x_train = x_train / 255
x_val /= 255

# y_data : 10 -> 2(one-hot)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# model 생성 
model = Sequential()

input_shape = (28, 28, 1)

# 1. CNN Model layer

# Convolution1 : [5,5,1,64]
model.add(Conv2D(64, kernel_size=(5,5), padding='same',
                 input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

# Convolution2 : [5,5,64,128]
model.add(Conv2D(128, kernel_size=(5,5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

# Flatten layer : 3d -> 1d
model.add(Flatten()) 

# Affine layer(Fully connected + relu) : [n, 256]
model.add(Dense(256, activation = 'relu'))

# Output layer(Fully connected + softmax) : [256, 10]
model.add(Dense(10, activation = 'softmax'))

# 2. model compile
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy', # one hot encoding
              metrics = ['accuracy'])

# 3. model train
model_fit = model.fit(x=x_train, y=y_train, 
                      epochs=3,
                      batch_size=100, 
                      verbose=1)

# 4. model evaluation
model.evaluate(x=x_val, y=y_val)
'''
- 4s 387us/sample - loss: 0.0116 - accuracy: 0.9925
'''


