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
import math

if __name__ == '__main__':

    
    torch.serialization.add_safe_globals([Conv, Conv2d, BatchNorm2d, SiLU, C2f, ModuleList, Bottleneck, SPPF, MaxPool2d, SimFusion_3in, dilation_block, Upsample, Concat])
    torch.serialization.add_safe_globals([dill._dill._load_type])
    torch.serialization.add_safe_globals([DetectionModel, Detect, DFL, IterableSimpleNamespace, v8DetectionLoss, BboxLoss])
    #torch.serialization.add_safe_globals([ultralytics.nn.tasks])
    torch.serialization.add_safe_globals([Sequential, BCEWithLogitsLoss, TaskAlignedAssigner])

    # Load a model
    model = YOLO("D:\\PreTrainedTest\\YOLO-TS\\weights\\GTSDB_updated.pt")  # or model = YOLO("./best.pt")
    
    # Train the model
    #model.train(data="./TT100K-2016.yaml", epochs=200, batch=48, imgsz=640, device='0,1,2,3')

    # Evaluate model performance on the validation set
    #metrics = model.val(data="D:\\PreTrainedTest\\YOLO-TS\\GTSDB.yaml", imgsz=640, batch=1, device='0')
    """classNames= ["Speed limit (20km/h)","Speed limit (30km/h)","Speed limit (50km/h)","Speed limit (60km/h)","Speed limit (70km/h)","Speed limit (80km/h)","End of speed limit (80km/h)",
                 "Speed limit (100km/h)","Speed limit (120km/h)","No passing","No passing for vehicles over 3.5 metric tons","Right-of-way at next intersection",  "Priority road",
                 "Yield","Stop","No vehicles","Vehicles over 3.5 metric tons prohibited","No entry","General caution","Dangerous curve to the left","Dangerous curve to the right",
                 "Double curve","Bumpy road","Slippery road","Road narrows on the right","Road work","Traffic signals","Pedestrians","Children crossing","Bicycles crossing","Beware of ice/snow",
                 "Wild animals crossing","End of all speed and passing limits","Turn right ahead","Turn left ahead","Ahead only","Go straight or right","Go straight or left","Keep right",
                 "Keep left","Roundabout mandatory","End of no passing","End of no passing by vehicles over 3.5 metric tons"]"""
    #classNames= [ "i2","i4","i5","il100","il60","il80","io","ip","p10","p11","p12","p19","p23","p26","p27","p3","p5","p6","pg","ph4","ph4.5","ph5","pl100","pl120","pl20","pl30","pl40","pl5","pl50","pl60","pl70","pl80","pm20","pm30","pm55","pn","pne","po","pr40","w13","w32","w55","w57","w59","wo"]
    while True:
        results = model("C:\\Users\\piyus\\Downloads\\stop-sign-2-1865228737.jpg", save=True, stream=True)
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
        
        break

    # Export the model to ONNX format
    #path = model.export(format="engine", device='0', half=True, opset=12)