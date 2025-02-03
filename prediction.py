import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load Pretrained DeepLabV3 Model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# Define Image Preprocessing Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Satellite Image (Change to your image path)
image_path = "C:/Users/admin/Desktop/New_method/googlearth/2023_05.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Run Inference
with torch.no_grad():
    output = model(input_tensor)["out"][0]  # Extract segmentation output

# Convert Output to Segmentation Mask
output_predictions = output.argmax(0).byte().cpu().numpy()  # Get class with highest probability

# Building Class in COCO is typically labeled as 2 (Adjust if needed)
building_mask = (output_predictions == 1).astype(np.uint8)  # 1 for buildings, 0 otherwise

# Overlay Mask on Original Image
overlay = np.array(image)
overlay[building_mask == 1] = [255, 0, 0]  # Color buildings in red

# Display Original & Segmented Image
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title("Original Satellite Image")
ax[0].axis("off")

ax[1].imshow(overlay)
ax[1].set_title("Buildings Detected (Red Overlay)")
ax[1].axis("off")

plt.show()
