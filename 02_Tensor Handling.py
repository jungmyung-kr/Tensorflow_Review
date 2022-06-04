"""
02_Tensor Handling
"""

'''
문) 다음 조건과 <출력 결과>를 참고하여 a와 b변수를 정의하고, 브로드캐스팅 연산을 
    수행한 결과를 출력하시오.
    <조건1> a변수 : list 이용 
    <조건2> b변수 : Variable() 이용
    <조건3> c변수 계산식 : c = a * b 
         -> multiply()이용 
    <조건4> a,b,c변수 결과 출력     

< 출력 결과 > 
a= [ 1.  2.  3.] 
b= [[ 0.123]     
    [ 0.234]
    [ 0.345]]
    
c= [[ 0.123       0.24600001  0.36900002]  : 3x3
   [ 0.234       0.46799999  0.70200002]
   [ 0.345       0.69        1.03499997]]
'''


import tensorflow as tf

a = [ 1.,  2.,  3.] # list 이용
b = tf.Variable([[ 0.123],[ 0.234],[ 0.345]])
c = tf.math.multiply(a, b) 
print('c =', c.numpy())
print(c.get_shape()) # (3, 3)


###############################################################################


'''
문2) 다음과 같이 X, a 행렬을 상수로 정의하고 행렬곱으로 연산하시오.
    단계1 : X, a 행렬 
        X 행렬 : iris 2~4번 칼럼으로 상수 정의 (tf.constant 이용)
        a 행렬 : [[0.2],[0.1],[0.3]] 값으로 상수 정의 (tf.constant 이용) 

    단계2 : 행렬곱 이용 y 계산하기  (tf.linalg.matmul 이용)

    단계3 : y 결과 출력
'''

import tensorflow as tf
import pandas as pd 

iris = pd.read_csv('C:\\ITWILL\\6_Tensorflow\\data\\iris.csv')
iris.info()

#  단계1 : X, a 상수 정의 

X = tf.constant(iris.iloc[:,1:4])
a = tf.constant( [[0.2],[0.1],[0.3]])

# 단계2 : 행렬곱 식 정의 

X.get_shape() # [150, 3]
a.get_shape() # [3, 1]

mat_mul = tf.linalg.matmul(a=X.numpy(), b=a.numpy()) 

# 단계3 : 행렬곱 결과 출력 
print(mat_mul.numpy())
mat_mul.get_shape() # TensorShape([150, 1])