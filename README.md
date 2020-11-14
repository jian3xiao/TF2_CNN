# TF2下的cnn实验

内容主要来自书《简明的TensorFlow 2》，[卷积神经网章节](https://tf.wiki/zh_hans/basic/models.html#cnn)

## 使用说明
1.  tf2_cnn.py 使用keras的子类化(Subclassing)API建立模型，即对 tf.keras.Model 类进行扩展以定义自己的新模型，同时手工编写了训练和评估模型的流程。这种方式灵活度高，且与其他流行的深度学习框架（如 PyTorch、Chainer）共通。

2. tf2_cnn_seq.py 使用keras的序列化(Sequential)API建立模型，通过向 tf.keras.models.Sequential() 提供一个层的列表，就能快速地建立一个 tf.keras.Model 模型。不过，这种层叠结构并不能表示任意的神经网络结构。

```
在上面的两个使用中tf2_cnn_seq.py 的训练速度更快！
```

3. tf2_cnn_dataset.py 是在 tf2_cnn.py 的基础上，使用tf.data.Dataset类来处理数据。当进行多线程处理时，确实比用 data_loader.get_batch(batch_size) 更快。.prefetch(4)=46.316秒，而使用.get_batch用时67.282秒。

4. tf2_cnn_dataset.py 是在**Eager模式**下的，对于每个batch数据的提取可用使用

```python
    for step, (image_batch, label_batch) in enumerate(mnist_dataset):  # 迭代数据集对象，带step参数

或者
    
    for image_batch, label_batch in mnist_dataset:
```

5. 当在**非Eager模式**下时，读取dataset类中batch的方法为。如果使用.repeat()方法设置epoch数，最好写成.repeat(epoch+1)，避免最后一个epoch报错。.repeat()在.shuffle()之后。

```python
    mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))\
    .shuffle(buffer_size=10000).batch(batch_size=batch_size).prefetch(4)
    iterator = mnist_dataset.make_one_shot_iterator()  # 从dataset中实例化一个Iterator
    image_batch, label_batch = iterator.get_next()
    
    sess = tf.Session()
    batch_images = sess.ren(image_batch)
```


```
以上实验设备：i7-9700K，2080Ti
```










