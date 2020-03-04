import tensorflow as tf
import os
import numpy as np 
import matplotlib.pyplot as plt 
import mdata

#0导入模块,生成模拟数据集
#喂入给定格子大小的数据学习
xs = mdata.l25
ys = mdata.ml25

xs1 = mdata.l20
ys1 = mdata.ml20

#设定所要预测m(kT)曲线格子的大小
xp0 = mdata.l20
xp1 = mdata.l19
xp2 = mdata.l27
xp3 = mdata.l18
xp4 = mdata.l30
xp5 = mdata.l28
xp6 = mdata.l26
#1定义神经网络的输入、参数和输出,定义前向传播过程
INPUT_NODE = 2		#输入层的神经元
OUTPUT_NODE = 1		#输出层的神经元
LAYER1_NODE = 500	#第一个隐藏层的神经元数

#定义权重w
def get_weight(shape,regularizer):
	w = tf.Variable(tf.random_normal(shape,stddev=0.1))
	if regularizer != None:tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

#定义偏置b
def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

#定义前向传播的过程
def forward(x,regularizer):

	w1 = get_weight([INPUT_NODE,LAYER1_NODE],regularizer)
	b1 = get_bias([LAYER1_NODE])
	y1 = tf.nn.sigmoid(tf.matmul(x,w1)+b1)

	w2 = get_weight([LAYER1_NODE,OUTPUT_NODE],regularizer)
	b2 = get_bias([OUTPUT_NODE])
	y = tf.nn.sigmoid(tf.matmul(y1,w2)+b2)
	return y


#2定义反向传播过程
learning_rate_base = 0.1	#初始学习率
learning_rate_decay = 0.99	#学习率衰减率
learning_rate_step = 1000	#喂入多少轮后开始更新一次学习率
regularizer = 0.0001		#正则化
steps = 2000			#训练次数
moving_average_decay = 0.99	#滑动平均衰减率
model_save_path="./model/"	#模型保存路径
model_name="Isingm_model"	#模型名称

#输入输出占位
x = tf.placeholder(tf.float32,[None,INPUT_NODE])
y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE])
y = forward(x,regularizer)
global_step = tf.Variable(0,trainable=False)	#计算训练轮数，不可训练

#定义损失函数
cem = tf.reduce_mean(tf.square(y - y_))
loss = cem + tf.add_n(tf.get_collection('losses'))

#定义指数下降学习率
learning_rate = tf.train.exponential_decay(
	learning_rate_base,
	global_step,
	learning_rate_step,
	learning_rate_decay,
	staircase=True)		#global_step/learning_rate_step取整数，阶梯型

#定义反向传播方法-梯度下降
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss, global_step=global_step)
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
#train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step)

#引入滑动平均,实例化滑动平均类
ema = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
#ema.apply后括号内为更新列表，运行sess.run（ema_op），更新滑动平均值
ema_op = ema.apply(tf.trainable_variables())
with tf.control_dependencies([train_step,ema_op]):
	train_op = tf.no_op(name='train')

saver = tf.train.Saver()

#3生成会话，训练STEPS轮
with tf.Session() as sess:
	init_op =tf.global_variables_initializer()
	sess.run(init_op)

	#断点寻回继续学习
	ckpt = tf.train.get_checkpoint_state(model_save_path)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)

	for i in range(steps):
		sess.run([train_op,loss,global_step],feed_dict={x: xs1, y_:ys1})
		_,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x: xs, y_:ys})
		if i % 1000 == 0:
			print("After %d training steps, loss is %g."%(step,loss_value))
		saver.save(sess, os.path.join(model_save_path,model_name),global_step=global_step)

#	y_pre = sess.run(y,{x:xs})
#	print(y_pre)
#	print(sess.run(y,{x:xp0}),sess.run(y,{x:xp1}),sess.run(y,{x:xp2}),sess.run(y,{x:xp3}),sess.run(y,{x:xp4}),sess.run(y,{x:xp5}))

	print(sess.run(y,{x:xp6}))






