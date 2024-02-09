import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Load and preprocess data (MNIST dataset)
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

strategy = tf.distribute.MirroredStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')

with strategy.scope():
    model = create_model()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

model.fit(train_dataset, epochs=5)

