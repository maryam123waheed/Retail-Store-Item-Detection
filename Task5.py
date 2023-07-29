import torch
from PIL import Image
import requests
from io import BytesIO

confidence_threshold = 0.5

def detect_retail_items(image_path):
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    results = model(image, size=640)

    class_ids = model.names
    bounding_boxes = results.pred[0][:, :4].tolist()
    confidences = results.pred[0][:, 4].tolist()
    class_names = [class_ids[int(cls)] for cls in results.pred[0][:, 5].tolist()]

    filtered_indices = results.pred[0][:, 4] > confidence_threshold
    bounding_boxes = results.pred[0][:, :4][filtered_indices].tolist()
    confidences = results.pred[0][:, 4][filtered_indices].tolist()
    class_names = [class_ids[int(cls)] for cls in results.pred[0][:, 5][filtered_indices].tolist()]

    detected_items = []
    for idx in range(len(class_names)):
        item = {
            'class_name': class_names[idx],
            'confidence': confidences[idx],
            'bounding_box': bounding_boxes[idx]
        }
        detected_items.append(item)

    return detected_items

if __name__ == '__main__':
    image_path = "C:/Users/marya/Downloads/Tasks data/grocery-store.jpg"
    detected_items = detect_retail_items(image_path)
    print(detected_items)
