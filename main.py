from keras.datasets import mnist
from keras import models
from keras import layers
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # loading images with labels

# create network
network = models.Sequential() 
network.add(layers.Dense(512,
                         activation='relu',
                         input_shape=(28 * 28,))) # First layer
network.add(layers.Dense(10,
                         activation='softmax')) # Loss layer

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])  # compile network

# Preparation of initial data
train_images = train_images.reshape((60000, 28 * 28)) 
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Preperation labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Start training
network.fit(train_images,
            train_labels,
            epochs=5,
            batch_size=128)

test_loss, test_acc = network.evaluate(test_images,
                                       test_labels) # Loss and accuracy
print(f'test_acc: {test_acc}') # Show accuracy
