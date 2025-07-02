# import tensorflow as tf
# from sensorflow_model.task import get_parameters
# from sensorflow_model.task import load_model  

# # Load the trained model
# model = load_model()

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# # Save the TFLite model
# with open("model.tflite", "wb") as f:
#     f.write(tflite_model) # type: ignore

# print("TFLite model exported as model.tflite")