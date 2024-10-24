from django.conf import settings
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def create_train_model():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Preprocess the data
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    # Build the model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_images, train_labels, epochs=100, validation_data=(test_images, test_labels))

    model_path = (os.path.join(settings.BASE_DIR, 'recognition_app', 'AI_model', 'trained_models', 'mnist_model.h5'))
    print(model_path)

    # Save the model
    model.save(model_path)

def load_train_model():
    model_path = (os.path.join(settings.BASE_DIR, 'recognition_app', 'AI_model', 'trained_models', 'mnist_model.h5'))
    print(model_path)

    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Evaluate the model
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

    return model

def predict(image):
    model = load_train_model()
    predictions = model.predict(image)
    for i in range(10):
        print(f'{i}: {predictions[0][i]}')
    predicted_class = np.argmax(predictions)
    return predicted_class;
