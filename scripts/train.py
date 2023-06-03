import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator

base_dir = '../training'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_not_photos_dir = os.path.join(train_dir, 'not-photos')
train_photos_dir = os.path.join(train_dir, 'photos')
validation_not_photos_dir = os.path.join(validation_dir, 'not-photos')
validation_photos_dir = os.path.join(validation_dir, 'photos')

model = tf.keras.models.Sequential([
    # since Conv2D is the first layer of the neural network, we should also specify the size of the input
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    # apply pooling
    tf.keras.layers.MaxPooling2D(2, 2),
    # and repeat the process
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # flatten the result to feed it to the dense layer
    tf.keras.layers.Flatten(),
    # and define 512 neurons for processing the output coming by the previous layers
    tf.keras.layers.Dense(512, activation='relu'),
    # a single output neuron. The result will be 0 if the image is a cat, 1 if it is a dog
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    optimizer="adam",
    loss='binary_crossentropy',
    metrics=['accuracy'])

# we rescale all our images with the rescale parameter
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# we use flow_from_directory to create a generator for training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150))

# we use flow_from_directory to create a generator for validation
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150))

history = model.fit(
    train_generator,  # pass in the training generator
    steps_per_epoch=100,
    epochs=15,
    validation_data=validation_generator,  # pass in the validation generator
    validation_steps=50,
    verbose=2)

model.save('../model')
