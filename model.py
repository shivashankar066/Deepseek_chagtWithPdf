import json
import cv2
import google.generativeai as genai
import numpy as np
import os

# Set up Gemini API key
os.environ["GOOGLE_API_KEY"] = ""
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the image
image_path = "C:/Users/admin/Desktop/New_method/googlearth/2024_03.jpg"
output_path = "C:/Users/admin/Desktop/New_method/googlearth/detected_buildings.jpg"
image = cv2.imread(image_path)

# Load image using PIL (Gemini requires PIL images)
from PIL import Image
img = Image.open(image_path)

# Initialize Gemini Pro Vision
model = genai.GenerativeModel("gemini-1.5-flash")

# Perform object detection
response = model.generate_content(
    [
        "Detect buildings in the image and return bounding box coordinates in JSON format.",
        img  # Pass the image here
    ]
)

# Debug: Print response
print("Full Response:", response.text)

# Try parsing JSON response
try:
    response_data = json.loads(response.text)

    if "objects" not in response_data:
        print("No 'objects' key found in response.")
    else:
        for obj in response_data["objects"]:
            if obj["name"].lower() == "building":
                for box in obj.get("bounding_boxes", []):
                    x, y, w, h = box["x"], box["y"], box["width"], box["height"]
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box

        # Save and display output image
        cv2.imwrite(output_path, image)
        cv2.imshow("Detected Buildings", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"Processed image saved at: {output_path}")

except json.JSONDecodeError as e:
    print("Error decoding JSON:", e)
