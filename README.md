# tf-tensorboard-dropout-demo
# TensorFlow可视化界面与过拟合

## TensorFlow可视化界面--tensorboard介绍

tensorflow提供了一个十分强大的功能--tensorboard可视化面板，我想这也是很多人选择使用tf的原因。有了它我们就可以很方便的查看模型训练过程中loss、accuracy、weight、biase、lr等等元素的变化过程。并且其会自动帮助我们绘制图表，大大简化我们训练模型的时间也有助于调参，也可以查看我们所设计的模型架构。而且只需要简单的几行代码就可以实现很炫酷的功能。首先给一个官网的[README连接](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/tensorboard/README.md)，可以先去看一下了解其用法和功能。TensorBoard大致界面如下图所示：

![](http://i.imgur.com/Wm1xiTz.png)

### TensorBoard可视化示例--mnist
#### 1，代码 ####
我们使用官网上面的[mnist示例](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py)来介绍tensorboard的基础用法，首先看一下其代码：

	from __future__ import absolute_import
	from __future__ import division
	from __future__ import print_function
	
	import argparse
	import sys
	
	import tensorflow as tf
	
	from tensorflow.examples.tutorials.mnist import input_data
	
	FLAGS = None
	
	
	def train():
	  # Import data
	  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)
	  sess = tf.InteractiveSession()
	
	  with tf.name_scope('input'):
	    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
	    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
	
	  with tf.name_scope('input_reshape'):
	    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
	    tf.summary.image('input', image_shaped_input, 10)
	
	  def weight_variable(shape):
	    initial = tf.truncated_normal(shape, stddev=0.1)
	    return tf.Variable(initial)
	
	  def bias_variable(shape):
	    initial = tf.constant(0.1, shape=shape)
	    return tf.Variable(initial)
	
	  def variable_summaries(var):
	    with tf.name_scope('summaries'):
	      mean = tf.reduce_mean(var)
	      tf.summary.scalar('mean', mean)
	      with tf.name_scope('stddev'):
	        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	      tf.summary.scalar('stddev', stddev)
	      tf.summary.scalar('max', tf.reduce_max(var))
	      tf.summary.scalar('min', tf.reduce_min(var))
	      tf.summary.histogram('histogram', var)
	
	  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
	    with tf.name_scope(layer_name):
	      with tf.name_scope('weights'):
	        weights = weight_variable([input_dim, output_dim])
	        variable_summaries(weights)
	      with tf.name_scope('biases'):
	        biases = bias_variable([output_dim])
	        variable_summaries(biases)
	      with tf.name_scope('Wx_plus_b'):
	        preactivate = tf.matmul(input_tensor, weights) + biases
	        tf.summary.histogram('pre_activations', preactivate)
	      activations = act(preactivate, name='activation')
	      tf.summary.histogram('activations', activations)
	      return activations
	
	  hidden1 = nn_layer(x, 784, 500, 'layer1')
	
	  with tf.name_scope('dropout'):
	    keep_prob = tf.placeholder(tf.float32)
	    tf.summary.scalar('dropout_keep_probability', keep_prob)
	    dropped = tf.nn.dropout(hidden1, keep_prob)
	
	  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)
	
	  with tf.name_scope('cross_entropy'):
	    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
	    with tf.name_scope('total'):
	      cross_entropy = tf.reduce_mean(diff)
	  tf.summary.scalar('cross_entropy', cross_entropy)
	
	  with tf.name_scope('train'):
	    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
	        cross_entropy)
	
	  with tf.name_scope('accuracy'):
	    with tf.name_scope('correct_prediction'):
	      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	    with tf.name_scope('accuracy'):
	      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	  tf.summary.scalar('accuracy', accuracy)
	
	  merged = tf.summary.merge_all()
	  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
	  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
	
	  tf.global_variables_initializer().run()
	
	  def feed_dict(train):
	    if train or FLAGS.fake_data:
	      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
	      k = FLAGS.dropout
	    else:
	      xs, ys = mnist.test.images, mnist.test.labels
	      k = 1.0
	    return {x: xs, y_: ys, keep_prob: k}
	
	  for i in range(FLAGS.max_steps):
	    if i % 10 == 0:  # Record summaries and test-set accuracy
	      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
	      test_writer.add_summary(summary, i)
	      print('Accuracy at step %s: %s' % (i, acc))
	    else:  # Record train set summaries, and train
	      if i % 100 == 99:  # Record execution stats
	        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	        run_metadata = tf.RunMetadata()
	        summary, _ = sess.run([merged, train_step],
	                              feed_dict=feed_dict(True),
	                              options=run_options,
	                              run_metadata=run_metadata)
	        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
	        train_writer.add_summary(summary, i)
	        print('Adding run metadata for', i)
	      else:  # Record a summary
	        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
	        train_writer.add_summary(summary, i)
	  train_writer.close()
	  test_writer.close()
	
	
	def main(_):
	  if tf.gfile.Exists(FLAGS.log_dir):
	    tf.gfile.DeleteRecursively(FLAGS.log_dir)
	  tf.gfile.MakeDirs(FLAGS.log_dir)
	  train()
	
	
	if __name__ == '__main__':
	  parser = argparse.ArgumentParser()
	  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
	                      default=False,
	                      help='If true, uses fake data for unit testing.')
	  parser.add_argument('--max_steps', type=int, default=1000,
	                      help='Number of steps to run trainer.')
	  parser.add_argument('--learning_rate', type=float, default=0.001,
	                      help='Initial learning rate')
	  parser.add_argument('--dropout', type=float, default=0.9,
	                      help='Keep probability for training dropout.')
	  parser.add_argument(
	      '--data_dir',
	      type=str,
	      default='/tmp/tensorflow/mnist/input_data',
	      help='Directory for storing input data')
	  parser.add_argument(
	      '--log_dir',
	      type=str,
	      default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
	      help='Summaries log directory')
	  FLAGS, unparsed = parser.parse_known_args()
	  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
#### 2，代码解释 ####
这里面只需要注意下面几个部分：

- variable_summary(var)函数：这个函数主要功能是对一个tensor，比如说权重/偏置/某一层的输出等，求其mean/min/max/stddev，并将相应的值（scalar）都保存到summary，并且将其本身保存到histogram中。
- 各个tensor的summary：每当你想要记录一个变量的变化时，就需要使用对应的summary，比如loss，accuracyd等。
- summary的merge：在记录完所有的tensor之后，接下来我们需要将其进行merge，这样的好处是我们在run的时候只需要指定最终merge变量即可。而不需要对每个summary进行run和写入文件的操作。
- 定义FileWriter：对于我们想要记录的summary我们需要写入文件才可以，FileWriter的作用就是定义文件的写入操作。
- add_summary：在每次执行sess.run()函数之后，我们都需要调用这个函数将最新的summary写入文件之中，这样我们才能时时更新每个变量的变化情况。


#### 3，最终的效果 ####
可以看一下其效果，在前端界面中主要包含下面几个选项：

![](http://i.imgur.com/bNbhfJH.png)

第一个我们要看的是GRAPHS，这里面是我们所构建的模型架构图，可以看一下是否正确，和我们想的是否一致。

![](http://i.imgur.com/5AIqmpE.png)

第二个我们要看的就是loss和accuracy等模型训练的指标是否符合预期，如下图所示，我们发现准确率在升高，loss在降低，而且训练集和测试集同步，也就是未发生过拟合现象，说明程序运行是正确的：

![](http://i.imgur.com/fhW5tIp.png)

第三个我们可以去看一下weight和biase的变化情况，这个有助于我们去观察模型训练是否正确，有没有出现异常情况等，这两个图一个是分布情况，一个是以直方图的形式展现数据的分布。

![](http://i.imgur.com/OL4Ryu0.png)
![](http://i.imgur.com/xHcPcOV.png)

第四个我们可以看一下所生成的图片效果，这个可以用来展示输入数据、卷积之后的图片等，可以更加直观的展示出每一层的意义和作用。

![](http://i.imgur.com/XaUb2wQ.png)
#### 4，函数功能总结 ####
接下来我们总结一下几个常用的函数相关用法，首先我们可以查看[官网的API](https://www.tensorflow.org/api_docs/python/tf/summary)进行了解，结合上面代码中的实际用法理解其输入输出意义：
![](http://i.imgur.com/6m5hWET.png)

- tf.summary.scalar(name, tensor, collections=None)：记录标量，如loss，accuracy，learning_rate等
- tf.summary.histogram(name, tensor, collections=None)：记录权重、输出、梯度的分布情况
- tf.summary.merge_all/tf.summary.merge(inputs, collections=None, name=None)：合并所有/指定变量
- tf.summary.FileWriter：写入summary文件
- Tf.summary.image(name, tensor, max_outputs=3, collections=None): 记录图片数据
- Tf.summary.audio(name, tensor, sample_rate, max_outputs=3, collections=None): 记录音频数据
- Tf.summary.text(name, tensor, collections=None): 记录文本数据

#### 5，可视化流程总结 ####
从上面可以看出来，tf的summary流程可以总结为下面几个步骤：

1. 针对每个我们想要观察的tensor，按照其类型（scalar，histogram，image，text，audio等）使用不同的tf.summary函数进行序列化；
2. 对训练/验证分别做merge，将想要展示的变量merge成一个op，然后使用tf.summary.FileWriter函数生成不同的summary writer用于将merge后的summary保存到日志文件中；
3. 在训练过程中可以隔几步调用一次add_summary将相关信息写入到日志文件中。
4. tensorboard --logdir='log_dir'，然后就可以在前端（localhost:6006）查看相关可视化信息了。

至此我们就了解了TensorBoard的工作原理和使用方法以及相关函数的作用。

### TensorBoard可视化示例--embedding ###

上面一个例子中我们介绍了mnist数据集在TensorBoard中进行可视化的相关知识，主要集中在显示训练过程中各个tensor值、分布情况以及image的可视化，还有Graph的可视化功能。那么接下来我们在介绍一个embedding可视化的例子。如果你还想接着对mnist数据进行相关的操作，可以参考下面这个链接，其效果图如下所示：

[使用TensorBoard对mnist进行embedding可视化](http://blog.csdn.net/aliceyangxi1987/article/details/71079387)

[动态效果视频](https://www.tensorflow.org/images/embedding-mnist.mp4)

![](http://i.imgur.com/zDOTnMm.png)

但是这里我们想做一个对word embedding进行可视化的例子。这里我们使用提前[与训练好的词向量](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.zh.vec)直接进行可视化操作。
代码很简单，就是将训练好的词向量加载进来，然后写入到log中即可，如下：
	import os
	import tensorflow as tf
	import numpy as np
	import word2vec
	from tensorflow.contrib.tensorboard.plugins import projector
	
	# load model
	model = word2vec.load('wiki.zh_classical.txt')
	embedding = model.vectors
	# setup a TensorFlow session
	tf.reset_default_graph()
	sess = tf.InteractiveSession()
	X = tf.Variable([0.0], name='embedding')
	place = tf.placeholder(tf.float32, shape=embedding.shape)
	set_x = tf.assign(X, place, validate_shape=False)
	sess.run(tf.global_variables_initializer())
	sess.run(set_x, feed_dict={place: embedding})
	
	# write labels
	with open('log/metadata.tsv', 'w', encoding='utf-8') as f:
	    for word in model.vocab:
	        f.write(word + '\n')
	
	# create a TensorFlow summary writer主要功能实现集中在这里，要想将想要的单词加入embedding，我们就需要config = projector.ProjectorConfig()
	#目的是建立一个链接，将词向量和上面生成的字典构成一个映射关系。这样生成出来的每个点才会对应到不同词。
	summary_writer = tf.summary.FileWriter('log', sess.graph)
	config = projector.ProjectorConfig()
	embedding_conf = config.embeddings.add()
	embedding_conf.tensor_name = 'embedding:0'
	embedding_conf.metadata_path = os.path.join('log', 'metadata.tsv')
	projector.visualize_embeddings(summary_writer, config)
	
	# save the model
	saver = tf.train.Saver()
	saver.save(sess, os.path.join('log', "model.ckpt"))

代码很简单，不做过多解释，下面我们直接看效果图：

![](http://i.imgur.com/hw0zqxN.png)

这张是embedding菜单栏下面的主图，会一直转动，当我们点击其中某个点时，会变成下面这样，显示出与被点单词距离最近的一些词语，方便我们查看最终的训练结果是否符合语义。比如我们点击“方言”，那么跟他语义相近的词都会被以红色加深表示，其他词则会变成灰色弱化。那么我们可以看出来，训练的效果还是很不错的。

![](http://i.imgur.com/BYKkQYf.png)

那么至此我们就完成了使用TensorBoard进行可视化的相关操作，基本上比较全面的掌握了TensorBoard的所有功能。

最后再推荐一个使用TensorBoard对char和word embedding进行可视化的项目，[https://github.com/Franck-Dernoncourt/NeuroNER](https://github.com/Franck-Dernoncourt/NeuroNER)，其效果图如下所示：
![](https://i.stack.imgur.com/SLXk1.jpg)

![](https://i.stack.imgur.com/U5y9j.png)

## 使用tensorflow解决过拟合问题 ##

### 过拟合简介 ###

过拟合问题可以说是机器学习和深度学习中最常见也很棘手的问题之一，一些基础的概念不再赘述，这里主要总结一下常见的用于解决过拟合现象的方法和手段：
![](http://campus.swarma.org/public/ueditor/php/upload/image/20170427/1493284525151436.png)

1. 正则化（参数范数惩罚L1,L2）
2. 数据增强（通过对样本进行各种操作：图片的形变、位移等，音频的声调、音量等、文本的反序）
3. 提前终止（early stopping）
4. dropout（在深度学习中最为常用的手段，其实就是随机将一部分神经元失活）
5. batch normalization（深度学习中对每一层的输入数据进行标准归一化操作）
6. bagging等模型集成的方法（通过组合多个模型减少泛化误差）

### 过拟合实例 ###

接下来我们要通过一个具体的例子来说明何为过拟合并且如何使用上述手段进行解决。我们使用mnist数据集进行演示，代码如下，构建了一个五层的神经网络，选择dropout作为减少过拟合的手段。运行两次代码，第一次令dropout=1.0，第二次令dropout=0.75。也就是说第一次的时候没有使用dropout，第二次则用了。下面看一下二者的效果：

	# encoding: UTF-8
	import tensorflow as tf
	import tensorboard.tensorflowvisu as tensorflowvisu
	import math
	from tensorflow.examples.tutorials.mnist import input_data as mnist_data
	print("Tensorflow version " + tf.__version__)
	tf.set_random_seed(0)
	
	mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
	
	X = tf.placeholder(tf.float32, [None, 28, 28, 1])
	Y_ = tf.placeholder(tf.float32, [None, 10])
	lr = tf.placeholder(tf.float32)
	pkeep = tf.placeholder(tf.float32)
	L = 200
	M = 100
	N = 60
	O = 30
	W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
	B1 = tf.Variable(tf.ones([L])/10)
	W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
	B2 = tf.Variable(tf.ones([M])/10)
	W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
	B3 = tf.Variable(tf.ones([N])/10)
	W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
	B4 = tf.Variable(tf.ones([O])/10)
	W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
	B5 = tf.Variable(tf.zeros([10]))
	
	XX = tf.reshape(X, [-1, 28*28])
	
	Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
	Y1d = tf.nn.dropout(Y1, pkeep)
	
	Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
	Y2d = tf.nn.dropout(Y2, pkeep)
	
	Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
	Y3d = tf.nn.dropout(Y3, pkeep)
	
	Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
	Y4d = tf.nn.dropout(Y4, pkeep)
	
	Ylogits = tf.matmul(Y4d, W5) + B5
	Y = tf.nn.softmax(Ylogits)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
	cross_entropy = tf.reduce_mean(cross_entropy)*100
	correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
	allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)
	I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
	It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
	datavis = tensorflowvisu.MnistDataVis()
	
	train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	
	def training_step(i, update_test_data, update_train_data):
	    batch_X, batch_Y = mnist.train.next_batch(100)
	
	    max_learning_rate = 0.003
	    min_learning_rate = 0.0001
	    decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
	    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
	
	    if update_train_data:
	        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], {X: batch_X, Y_: batch_Y, pkeep: 1.0})
	        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
	        datavis.append_training_curves_data(i, a, c)
	        datavis.update_image1(im)
	        datavis.append_data_histograms(i, w, b)
	
	    if update_test_data:
	        a, c, im = sess.run([accuracy, cross_entropy, It], {X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
	        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
	        datavis.append_test_curves_data(i, a, c)
	        datavis.update_image2(im)
	    sess.run(train_step, {X: batch_X, Y_: batch_Y, pkeep: 1.0, lr: learning_rate})
	
	datavis.animate(training_step, iterations=10000+1, train_data_update_freq=20, test_data_update_freq=100, more_tests_at_start=True)
	
	print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

![](http://i.imgur.com/2l37V4u.png)

![](http://i.imgur.com/OG1lSBb.png)

![](http://i.imgur.com/TPCIGRY.png)


上面第一张图是没有使用dropout的，第二张图是准确度的输出结果，第三张图是使用了dropout的。可以看出在没有使用dropout的时候训练集准确度一直保持在1.0左右，图中蓝色的线都已经和1.0重合了，loss也基本上接近于0了，重要的是橙色曲线已经呈现出一种上升趋势，这说明训练集上数据仍然在变好，但是测试机上的效果在变差，这说明模型已经开始出现过拟合现象，模型的泛化能力逐渐变弱。
再看图三，明显感觉蓝色曲线（训练集）一直波动得很厉害，不管accuracy还是loss都没有出现上述情况，而且在测试集上（橙色曲线）也未出现上升现象，这说明dropout很好的减轻了模型的过拟合问题。但是从准确度和loss的角度上，也和没有使用dropout的时候保持相近的状态，acc都达到了0.98以上。

那么到这里我们也学习了如何使用dropout来解决模型的过拟合问题。总结一下就是在每一层（可以有选择的，不一定每层都加）激活函数之后加上一个dropout层，来随机的失活一定比例神经元以达到减轻过拟合的效果。

	Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
	Y1d = tf.nn.dropout(Y1, pkeep)
	
	Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
	Y2d = tf.nn.dropout(Y2, pkeep)
	
	Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
	Y3d = tf.nn.dropout(Y3, pkeep)
	
	Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
	Y4d = tf.nn.dropout(Y4, pkeep)

参考连接：

1，tensorflow官方的doc：[https://www.tensorflow.org/get_started/summaries_and_tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)

2，TensorFlow官网的API：[https://www.tensorflow.org/api_docs/python/tf/summary/scalar](https://www.tensorflow.org/api_docs/python/tf/summary/scalar)

3，colah大神的两篇可视化的博客：[http://colah.github.io/posts/2015-01-Visualizing-Representations/](http://colah.github.io/posts/2015-01-Visualizing-Representations/)

[http://colah.github.io/posts/2014-10-Visualizing-MNIST/](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)

4，机器之心的文章：[没有博士学位如何玩转TensorFlow和深度学习](https://www.jiqizhixin.com/articles/9cb76fe4-99c1-47a7-84d2-63c35c90d313)

5，命名实体识别项目：[https://github.com/Franck-Dernoncourt/NeuroNER](https://github.com/Franck-Dernoncourt/NeuroNER)

6，Facebook的fasttext项目预训练好的词向量：[https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.zh.vec](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.zh.vec)

