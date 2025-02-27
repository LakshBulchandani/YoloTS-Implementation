from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
#from models.experimental import attempt_load
from torchvision.ops import nms
import pathlib
import dlib
from scipy.spatial import distance
from ultralytics import YOLO
import math
import torch
import ultralytics
import cv2
from ultralytics import YOLO
import ultralytics
import torch
import dill
import torch.serialization
from torch.nn.modules.container import Sequential
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv, Concat
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF, DFL
from torch.nn.modules.container import ModuleList
from torch.nn.modules.pooling import MaxPool2d
from ultralytics.nn.modules import SimFusion_3in, dilation_block
from torch.nn.modules.upsampling import Upsample
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner
from torch.nn.modules.loss import BCEWithLogitsLoss
app = Flask(__name__)

torch.serialization.add_safe_globals([Conv, Conv2d, BatchNorm2d, SiLU, C2f, ModuleList, Bottleneck, SPPF, MaxPool2d, SimFusion_3in, dilation_block, Upsample, Concat])
torch.serialization.add_safe_globals([dill._dill._load_type])
torch.serialization.add_safe_globals([DetectionModel, Detect, DFL, IterableSimpleNamespace, v8DetectionLoss, BboxLoss])
#torch.serialization.add_safe_globals([ultralytics.nn.tasks])
torch.serialization.add_safe_globals([Sequential, BCEWithLogitsLoss, TaskAlignedAssigner, nms])

# Load YOLOv5 model
"""def load_traffic_sign_model():
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    model = attempt_load('weights/GTSDB.pt')  # Load model on gPU
    pathlib.PosixPath = temp
    return model"""

#model = load_traffic_sign_model()
model = YOLO("D:\\PreTrainedTest\\YOLO-TS\\weights\\GTSDB_updated.pt")

# Initialize face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

@app.route('/')
def home():
    return "Upload an image file on /detect for traffic sign detection or use /drowsiness for drowsiness detection"

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(img, stream=True)

    detections = process_results(results)
    return jsonify(detections)
def process_results(results):
    detections=[]
    for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", model.names[cls])

                print("x1,x2,y1,y2 --> ",x1,x2,y1,y2)

                detections.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "confidence": confidence,
                "class_id": int(cls),
                "class_name": model.names[int(cls)] if hasattr(model, 'names') else "Unknown"
        })
   
    return detections
                

"""@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        results = model(img_tensor)[0]
    
    detections = process_results(results)
    return jsonify(detections)

def preprocess_image(img):
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    return img

def process_results(results):
    conf_threshold = 0.25  
    nms_threshold = 0.45   

    results = results.squeeze(0)
    conf_scores = results[:, 4]
    valid_mask = conf_scores > conf_threshold
    filtered_results = results[valid_mask]

    if filtered_results.shape[0] == 0:
        return []

    boxes = filtered_results[:, :4]
    scores = filtered_results[:, 4]
    class_probs = filtered_results[:, 5:]

    class_ids = class_probs.argmax(dim=1)
    boxes[:, 2:] += boxes[:, :2]

    keep_indices = nms(boxes, scores, nms_threshold)
    boxes, scores, class_ids = boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]

    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        cls = class_ids[i]

        detections.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "confidence": float(conf),
            "class_id": int(cls),
            "class_name": model.names[int(cls)] if hasattr(model, 'names') else "Unknown"
        })
   
    return detections"""

@app.route('/drowsiness', methods=['POST'])
def drowsiness_detection():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_scale)

    drowsiness_results = []
    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))

        right_Eye = Detect_Eye(rightEye)
        left_Eye = Detect_Eye(leftEye)
        Eye_Rat = (left_Eye + right_Eye) / 2
        Eye_Rat = round(Eye_Rat, 2)

        if Eye_Rat < 0.25:
            drowsiness_results.append({
                "drowsiness_detected": True,
                "eye_aspect_ratio": Eye_Rat,
                "message": "Drowsiness Detected. Stop driving to prevent accidents."
            })
        else:
            drowsiness_results.append({
                "drowsiness_detected": False,
                "eye_aspect_ratio": Eye_Rat,
                "message": "No drowsiness detected."
            })

    return jsonify(drowsiness_results)

def Detect_Eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_Eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_Eye

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)