import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('../model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model.
open('../model.tflite', 'wb').write(tflite_model)