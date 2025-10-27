import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("resnet50.onnx")
x = np.random.randn(300, 3, 224, 224).astype(np.float32)
y = session.run(None, {"input": x})[0]
print("Output shape:", y.shape)
