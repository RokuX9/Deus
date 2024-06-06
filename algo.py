# import cv2 as cv
# img = cv.imread("path/to/image")

# cv.imshow("Display window", img)
# k = cv.waitKey(0) # Wait for a keystroke in the window

# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# # Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format

# from ultralytics import YOLO
# import  cv2
# import cvzone
# import math

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

# model = YOLO('../YOLO Weights/yolov8n.pt')

# classNames = ["person",
# "bicycle",
# "car",
# "motorbike",
# "aeroplane",
# "bus",
# "train",
# "truck",
# "boat",
# "traffic light",
# "fire hydrant",
# "stop sign",
# "parking meter",
# "bench",
# "bird",
# "cat",
# "dog",
# "horse",
# "sheep",
# "cow",
# "elephant",
# "bear",
# "zebra",
# "giraffe",
# "backpack",
# "umbrella",
# "handbag",
# "tie",
# "suitcase",
# "frisbee",
# "skis",
# "snowboard",
# "sports ball",
# "kite",
# "baseball bat",
# "baseball glove",
# "skateboard",
# "surfboard",
# "tennis racket",
# "bottle",
# "wine glass",
# "cup",
# "fork",
# "knife",
# "spoon",
# "bowl",
# "banana",
# "apple",
# "sandwich",
# "orange",
# "broccoli",
# "carrot",
# "hot dog",
# "pizza",
# "donut",
# "cake",
# "chair",
# "sofa",
# "pottedplant",
# "bed",
# "diningtable",
# "toilet",
# "tvmonitor",
# "laptop",
# "mouse",
# "remote",
# "keyboard",
# "cell phone",
# "microwave",
# "oven",
# "toaster",
# "sink",
# "refrigerator",
# "book",
# "clock",
# "vase",
# "scissors",
# "teddy bear",
# "hair drier",
# "toothbrush"]

# while True:
#     success, img  =cap.read()
#     results = model(img, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             w, h = x2-x1, y2-y1
#             cvzone.cornerRect(img, (x1, y1, w, h))

#             conf = math.ceil((box.conf[0]*100))/100

#             cls = box.cls[0]
#             name = classNames[int(cls)]

#             cvzone.putTextRect(img, f'{name} 'f'{conf}', (max(0,x1), max(35,y1)), scale = 0.5)



#     cv2.imshow("Image", img)
#     cv2.waitKey(1)


# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
# import numpy as np
# import mss

# # Initialize MSS for screen capture
# sct = mss.mss()

# # Define the part of the screen to capture
# monitor = sct.monitors[1]  # Capture the primary monitor

# model = YOLO('../YOLO Weights/yolov8n.pt')

# classNames = ["person",
# "bicycle",
# "car",
# "motorbike",
# "aeroplane",
# "bus",
# "train",
# "truck",
# "boat",
# "traffic light",
# "fire hydrant",
# "stop sign",
# "parking meter",
# "bench",
# "bird",
# "cat",
# "dog",
# "horse",
# "sheep",
# "cow",
# "elephant",
# "bear",
# "zebra",
# "giraffe",
# "backpack",
# "umbrella",
# "handbag",
# "tie",
# "suitcase",
# "frisbee",
# "skis",
# "snowboard",
# "sports ball",
# "kite",
# "baseball bat",
# "baseball glove",
# "skateboard",
# "surfboard",
# "tennis racket",
# "bottle",
# "wine glass",
# "cup",
# "fork",
# "knife",
# "spoon",
# "bowl",
# "banana",
# "apple",
# "sandwich",
# "orange",
# "broccoli",
# "carrot",
# "hot dog",
# "pizza",
# "donut",
# "cake",
# "chair",
# "sofa",
# "pottedplant",
# "bed",
# "diningtable",
# "toilet",
# "tvmonitor",
# "laptop",
# "mouse",
# "remote",
# "keyboard",
# "cell phone",
# "microwave",
# "oven",
# "toaster",
# "sink",
# "refrigerator",
# "book",
# "clock",
# "vase",
# "scissors",
# "teddy bear",
# "hair drier",
# "toothbrush"]

# while True:
#     # Capture the screen
#     screen_img = np.array(sct.grab(monitor))

#     # Convert the image from BGRA to BGR
#     screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)

#     # Perform object detection
#     results = model(screen_img, stream=True)

#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             w, h = x2 - x1, y2 - y1
#             cvzone.cornerRect(screen_img, (x1, y1, w, h))

#             conf = math.ceil((box.conf[0] * 100)) / 100

#             cls = box.cls[0]
#             name = classNames[int(cls)]

#             cvzone.putTextRect(screen_img, f'{name} {conf}', (max(0, x1), max(35, y1)), scale=0.5)

#     # Display the result
#     cv2.imshow("Screen Capture", screen_img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import mss

# Initialize MSS for screen capture
sct = mss.mss()

# Define the part of the screen to capture
monitor = sct.monitors[1]  # Capture the primary monitor

# Load the YOLO model
model = YOLO('../YOLO Weights/yolov8n.pt')

classNames = ["cursor"]

# Capture a single frame from the screen
screen_img = np.array(sct.grab(monitor))

# Convert the image from BGRA to BGR
screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)

# Perform object detection
results = model(screen_img)

# Process the results
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(screen_img, (x1, y1, w, h))

        conf = math.ceil((box.conf[0] * 100)) / 100

        cls = box.cls[0]
        name = classNames[int(cls)]

        cvzone.putTextRect(screen_img, f'{name} {conf}', (max(0, x1), max(35, y1)), scale=0.5)

# Display the result
cv2.imshow("Screen Capture", screen_img)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()
