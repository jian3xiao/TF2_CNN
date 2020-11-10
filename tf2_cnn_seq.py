# 来自 书《简明的TensorFlow 2》
import tensorflow as tf  # 2.1
import numpy as np
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)  # 设置按需要分配GPU资源


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


def CNN():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5],  padding='same',  activation=tf.nn.relu,
                                     input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2))
    model.add(tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,)))
    model.add(tf.keras.layers.Dense(units=1024, activation=tf.nn.relu, use_bias=False))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    return model


num_epochs = 5
batch_size = 50
learning_rate = 0.001
model = CNN()

data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)
test_loss, test_acc = model.evaluate(data_loader.test_data, data_loader.test_label)
print("test accuracy= %f, test_loss= %f" % (test_acc, test_loss))

