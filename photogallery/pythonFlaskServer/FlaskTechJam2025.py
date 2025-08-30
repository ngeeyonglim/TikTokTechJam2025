from flask import Flask, request
from ultralytics import YOLO
import cv2
import json

app = Flask(__name__)
model = YOLO("yolov8n.pt")

to_blur = ["person", "face"]


@app.route('/')
def home():
    return "Welcome to the Flask Tech Jam 2025!"


@app.route('/detect', methods=['POST'])
def detect():
    print("I have a request at /detect!")
    data = request.get_json()
    print(data)
    source = data.get("image")
    print(source, type(source))
    label = data.get("label")
    img = cv2.imread(source)
    h, w = img.shape[:2]
    results = model(img)[0]
    detections = []

    boxes = results.boxes.xyxy.tolist()

    class_ids = results.boxes.cls.tolist()   # Class indices

    for box, class_id in zip(boxes, class_ids):
        # Don't filter by label yet, return all 
        print("class: ", class_id)
        print("Hey returning!")
        x1, y1, x2, y2 = map(float, box)
        detections.append({
            "class_id": int(class_id),
            "bbox": [x1 / w,
                 y1 / h,
                 x2 / w,
                 y2 / h]
        })
            
        """
        if label == int(class_id):
            print("Hey returning!")
            x1, y1, x2, y2 = map(float, box)
            detections.append({
                "class_id": int(class_id),
                "bbox": [x1 / w,
                         y1 / h,
                         x2 / w,
                         y2 / h]
            })
            """

    json_output = json.dumps(detections, indent=2)
    print(f"im returning ${json_output}")
    return json_output


@app.route('/update_data', methods=['POST'])
def update_data():
    print("I have a request at /update_data!")
    data = request.get_json()
    print(data)
    labels = data.get("labels")
    bbox = data.get("new_bbs")
    path = data.get("image")

    if labels is None or bbox is None or path is None:
        return {"error": "Missing required fields"}, 400
        
    with open("new_training_data.txt", mode="w+") as td:
        for idx, label in enumerate(labels):
            entry = f"{label} {bbox[idx][0]} {bbox[idx][1]} {bbox[idx][2]} {bbox[idx][3]}\n"
            td.write(entry)

    # Update txt file containing bounding box information
    return {"success": True}, 201


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
