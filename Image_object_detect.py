import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Define transformation for input images
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function to detect objects in an image
def detect_objects(model, image, threshold=0.5):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    detected_objects = []
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= threshold:
            detected_objects.append({
                'box': box.tolist(),
                'label': label.item(),
                'score': score.item()
            })
    return detected_objects


# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, objects, label_map):
    draw = ImageDraw.Draw(image)
    for obj in objects:
        box = obj['box']
        label = label_map.get(obj['label'], f"Unknown ({obj['label']})")  # Handle unknown labels
        draw.rectangle(box, outline='red', width=10)
        draw.text((box[0], box[1] - 10), label, fill='red')
    return image


# Main function
def main():
    # Load pre-trained Faster R-CNN model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()


    label_map = {i: label for i, label in enumerate([
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'building'
    ])}

    # Path to the latest (2024) image
    image_2024_path = 'C:/Users/admin/Desktop/New_method/googlearth/2022_01.jpg'
      # 'C:/Users/admin/Desktop/New_method/googlearth/2023_05.jpg'   2022_01   2018_01
    # Load the image
    image_2024 = Image.open(image_2024_path).convert('RGB')

    # Detect objects in the image


    objects_2024 = detect_objects(model, image_2024)

    # Draw bounding boxes on the image
    image_2024_with_boxes = draw_bounding_boxes(image_2024.copy(), objects_2024, label_map)

    # Display the image with detected objects
    plt.figure(figsize=(8, 6))
    plt.imshow(image_2024_with_boxes)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
