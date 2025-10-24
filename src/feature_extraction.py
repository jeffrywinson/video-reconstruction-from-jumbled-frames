import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import cv2

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
        self.features = output.squeeze()

    def extract(self, frame_bgr):
        """
        Extracts features from a single BGR frame (from OpenCV).
        """
        # Convert from OpenCV's BGR to PIL's RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Apply transforms and add a batch dimension (B, C, H, W)
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        # Run the model. The hook will automatically store the features.
        with torch.no_grad():
            self.model(img_t)
            
        # Return features as a numpy array
        return self.features.cpu().numpy()