import json
import cv2
import ollama
import numpy as np
# Load the image
image_path = 'C:/Users/admin/Desktop/New_method/googlearth/2024_03.jpg'
output_path = 'C:/Users/admin/Desktop/New_method/googlearth/detected_buildings.jpg'
image = cv2.imread(image_path)
# Perform object detection using Ollama
res = ollama.chat(
    model="llama3.2-vision",
    messages=[
        {
            'role': 'user',
            'content': """Detect buildings in the image and return bounding box coordinates in JSON format.
            Provide output as:
            {
                "objects": [
                    {
                        "name": "Building",
                        "count": 5,
                        "color": ["Gray", "White"],
                        "bounding_boxes": [
                            {"x": 100, "y": 200, "width": 50, "height": 60},
                            {"x": 300, "y": 400, "width": 80, "height": 100}
                        ]
                    }
                ]
            }
            """,
            'images': [image_path]
        }
    ],
    format="json",
    options={'temperature': 0}
)

# Debug: Print response before parsing
print("Full Response:", res['message']['content'])

# Try parsing JSON response
try:
    response_data = json.loads(res['message']['content'])

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
