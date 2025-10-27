# export_model.py (Updated)
import torch
import torch.nn as nn
import torchvision.models as models
import onnxscript # Ensure this is installed: pip install onnxscript

print("Loading pre-trained ResNet-50...")

class ResNet50Features(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Include layers up to and including avgpool (output shape: [batch, 2048, 1, 1])
        self.features = nn.Sequential(*list(base.children())[:-1])
        # REMOVED: self.flatten = nn.Flatten(1)

    def forward(self, x):
        x = self.features(x)
        # Output is now [batch, 2048, 1, 1]
        return x

model = ResNet50Features().eval()

# Use a dummy input with batch > 1 to help exporter trace dynamic axes
dummy_input = torch.randn(2, 3, 224, 224) # Batch size 2
input_names = ["input"]
output_names = ["features"] # Output is now the pooled features

print("Exporting to resnet50.onnx (opset 18)...")

torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        'input': {0: 'batch'},    # Mark input batch dim as dynamic
        'features': {0: 'batch'} # Mark output batch dim as dynamic
    },
    opset_version=18
)

print("\nSuccess! 'resnet50.onnx' (output shape [batch, 2048, 1, 1]) has been created.")
print("Copy this new file into your 'jumbled_frames_rust' project folder, replacing the old one.")