# TF2下的cnn实验

内容主要来自书《简明的TensorFlow 2》，[卷积神经网章节](https://tf.wiki/zh_hans/basic/models.html#cnn)

## 使用说明
1.  tf2_cnn.py 使用keras的子类化(Subclassing)API建立模型，即对 tf.keras.Model 类进行扩展以定义自己的新模型，同时手工编写了训练和评估模型的流程。这种方式灵活度高，且与其他流行的深度学习框架（如 PyTorch、Chainer）共通。

2. tf2_cnn_seq.py 使用keras的序列化(Sequential)API建立模型，通过向 tf.keras.models.Sequential() 提供一个层的列表，就能快速地建立一个 tf.keras.Model 模型。不过，这种层叠结构并不能表示任意的神经网络结构。

'''
在上面的两个使用中tf2_cnn_seq.py 的训练速度更快！
'''










