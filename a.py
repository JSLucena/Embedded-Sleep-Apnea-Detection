import tensorflow as tf

# Load your TFLite model
model_path = 'models/dynamic.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input dtype:", input_details[0]['dtype'])  # Should be int8 if fully quantized
print("Output dtype:", output_details[0]['dtype'])  # Should be int8 if fully quantized
signature_list = interpreter.get_signature_list()  # (Optional) Check signatures if using multi-input models

# Get the list of all ops used in the model
op_list = set()
for op in interpreter._get_ops_details():
    op_list.add(op['op_name'])

print("Operators used in the model:")
print(op_list)