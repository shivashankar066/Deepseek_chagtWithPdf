import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision import models

# Load image (if it's a normal PNG or JPG)
image = cv2.imread("C:/Users/admin/Desktop/New_method/googlearth/2024_03.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Display the image
plt.imshow(image)
plt.title("Original Satellite Image")
plt.show()

import torch
import torchvision.transforms as transforms
from torchvision import models

# Load pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Preprocess the image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Run inference
with torch.no_grad():
    output = model(image_tensor)['out'][0]

# Convert to binary mask (assuming 'building' is class index 1 or similar)
output_mask = output.argmax(0).byte().cpu().numpy()

# Display the detected buildings
plt.imshow(output_mask, cmap="gray")
plt.title("Detected Buildings")
plt.show()

