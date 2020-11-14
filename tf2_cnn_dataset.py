# 来自 书《简明的TensorFlow 2》。 在tf2_cnn.py的基础上，使用tf.data.Dataset类来处理数据集
import tensorflow as tf  # 2.1
import numpy as np
import time
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)  # 设置按需要分配GPU资源


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu, use_bias=False)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


class MNISTLoader:
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]


num_epochs = 5
batch_size = 50
learning_rate = 0.001
model = CNN()

data_loader = MNISTLoader()
(train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data()
train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis=-1)
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))\
    .shuffle(buffer_size=10000).batch(batch_size=batch_size).prefetch(4)  # tf.data.experimental.AUTOTUNE

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

start = time.time()
for epoch in range(num_epochs):
    # batch_index = 1
    for image_batch, label_batch in mnist_dataset:  # 不使用.repeat(num_epochs)时，就是一次全部数据的所有，60000 / 50 = 1200
        with tf.GradientTape() as tape:
            y_pred = model(image_batch)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=label_batch, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            # print("[epoch %d][batch %d]: loss %f" % (epoch, batch_index, loss.numpy()))
            # batch_index = batch_index + 1
        grads = tape.gradient(loss, model.variables)  # 损失值和参数，计算梯度
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))  # 梯度和参数组成元组
print(time.time()-start)  # 66.380=AUTOTUNE,  47.065=prefetch(2), 46.316=.prefetch(4), 46.320=prefetch(6), 47.122=prefetch(10)
model.summary()


# num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
# start = time.time()
# for batch_index in range(num_batches):
#     X, y = data_loader.get_batch(batch_size)
#     with tf.GradientTape() as tape:
#         y_pred = model(X)
#         loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
#         loss = tf.reduce_mean(loss)
#         # print("batch %d: loss %f" % (batch_index, loss.numpy()))
#     grads = tape.gradient(loss, model.variables)  # 损失值和参数，计算梯度
#     optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))  # 梯度和参数组成元组
# print(time.time()-start)  # 67.282
# model.summary()

# test
# sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
# num_batches = int(data_loader.num_test_data // batch_size)
# for batch_index in range(num_batches):
#     start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
#     y_pred = model.predict(data_loader.test_data[start_index: end_index])
#     sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
# print("test accuracy: %f" % sparse_categorical_accuracy.result())

