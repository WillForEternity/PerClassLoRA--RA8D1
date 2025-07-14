import onnx
import os

# Construct the absolute path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.onnx')

# Load the ONNX model
model = onnx.load(MODEL_PATH)

# Get the input and output names
input_names = [input.name for input in model.graph.input]
output_names = [output.name for output in model.graph.output]

print(f"Model: {os.path.abspath(MODEL_PATH)}")
print(f"Input Names: {input_names}")
print(f"Output Names: {output_names}")
