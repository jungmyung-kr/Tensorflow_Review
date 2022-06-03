"""
01_Basic.py
"""

'''
문1) 두 상수를 정의하고, 사칙연산(+,-,*,/)을 정의하여 결과를 출력하시오.
  조건1> 두 상수 이름 : a, b
  조건2> 변수 이름 : adder,subtract,multiply,divide
  조건3> 출력 : 출력결과 예시 참고
  
<<출력결과>>
a= 100
b= 20
===============
덧셈 = 120
뺄셈 = 80
곱셈 = 2000
나눗셈 = 5.0
'''

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함 

'''프로그램 정의 영역'''

# 상수 정의 
a = tf.constant(100)
b = tf.constant(20) 

# 식 정의 
adder = a + b

# 변수 정의 
subtract = tf.Variable(a - b)
multiply = tf.Variable(a * b)
divide = tf.Variable(a / b)


'''프로그램 실행 영역'''
sess = tf.Session() # # session object 생성 

# 상수 장치 할당 
print('a=', sess.run(a))
print('b=', sess.run(b))
print('='*20)

# 변수 초기화 
sess.run(tf.global_variables_initializer()) # 변수 초기화

# 식 장치 할당(연산)
print('덧셈 = ', sess.run(adder))
print('뺄셈 = ', sess.run(subtract))
print('곱셈 = ', sess.run(multiply))
print('나눗셈 = ', sess.run(divide))

sess.close() # session 닫기 


###############################################################################


'''
문2) 다음과 같은 상수와 사칙연산 함수를 이용하여 dataflow의 graph를 작성하여 
    tensorboard로 출력하시오.
    조건1> 상수 : x = 100, y = 50
    조건2> 계산식 : result = ((x - 5) * y) / (y + 20)
       -> 사칙연산 함수 이용 계산식 작성  
        1. sub = (x - 5) : tf.subtract(x, 5)
        2. mul = ((x - 5) * y) : tf.multiply(sub, y)
        3. add = (y + 20) : tf.add(y, 20)
        4. div = mul / add : tf.div(mul, add)
   조건3> 출력 graph : 첨부파일 참고      
'''
import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함 

# tensorboard 초기화 
tf.reset_default_graph()


# 상수 정의 
x = tf.constant(100, name = 'x')
y = tf.constant(50, name = 'y')

# 계산식 정의 

sub = tf.subtract(x, 5, name = 'sub')
mul = tf.multiply(sub, y, name = 'mul')
add = tf.add(y, 20, name = 'add')
div = tf.div(mul, add, name = 'div')

with tf.Session() as sess :
    print('sub=', sess.run(sub))
    print('mul=', sess.run(mul))
    print('add=', sess.run(add))
    print('div=', sess.run(div))
    
    # tensorboard graph 생성
    tf.summary.merge_all() 
    writer = tf.summary.FileWriter(r'C:/ITWILL/6_Tensorflow/graph', sess.graph)
    writer.close()

'''
sub= 95
mul= 4750
add= 70
div= 67
'''







