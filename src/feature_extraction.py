import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import cv2
from tqdm import tqdm

class FeatureExtractor:
    """
    Uses a pre-trained ResNet-50 to extract deep feature vectors from frames.
    This is much more robust than comparing raw pixels.
    """
    def __init__(self):
        # Check for GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load a pre-trained ResNet-50
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
        self.model.eval() # Set to evaluation mode (disables dropout, etc.)

        # We'll grab the features from the 'avgpool' layer, right before the
        # final classification layer. This is a 2048-dim vector.
        self.feature_layer = self.model.avgpool
        self.features = None
        self.feature_layer.register_forward_hook(self.hook_fn)

        # Standard ImageNet pre-processing
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def hook_fn(self, module, input, output):
        """A 'hook' to capture the output of the avgpool layer"""
        # Squeeze the spatial dimensions, but keep the batch dimension
        self.features = output.squeeze(-1).squeeze(-1)

    def _preprocess_frame(self, frame_bgr):
        """
        Helper function to pre-process a single BGR frame into a
        transformed tensor.
        """
        # Convert from OpenCV's BGR to PIL's RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Apply transforms and return the tensor
        return self.transform(img)

    def extract_batch(self, frames):
        """
        Extracts features from a BATCH of BGR frames (from OpenCV).
        This is *much* faster than one by one.
        """
        print("Pre-processing frames for batch...")
        
        # Create a list of pre-processed frame tensors
        # This part is still sequential, but it's very fast.
        tensor_list = [self._preprocess_frame(f) for f in tqdm(frames, desc="Pre-processing")]
        
        # Stack all tensors into a single "batch" tensor (e.g., [300, 3, 224, 224])
        batch_tensor = torch.stack(tensor_list).to(self.device)
        
        print(f"Running batch of {len(frames)} frames through the model (this is the fast part)...")
        
        # Run the model ONCE on the entire batch.
        with torch.no_grad():
            self.model(batch_tensor)
            
        # The hook will have saved the [300, 2048] tensor.
        # Return it as a numpy array.
        return self.features.cpu().numpy()