# Import libs
import cv2
import numpy as np

#Load yolo
net = cv2.dnn.readNet('/Users/nravinuthala/temp/yolov3.weights','/Users/nravinuthala/temp/yolov3.cfg')
classes = []

with open('/Users/nravinuthala/temp/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    box_colors = np.random.uniform(0, 255, size=(len(classes), 3))
    label_colors = np.random.uniform(0, 255, size=(len(classes), 3))
#print(classes)

layer_names = net.getLayerNames()
unconn_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i[0] - 1] for i in unconn_layers]
#print(output_layers)

#img = cv2.imread('/Users/nravinuthala/temp/room_ser.jpg')
#img = cv2.imread('/Users/nravinuthala/temp/nag_fridge.jpeg')
#img = cv2.imread('/Users/nravinuthala/temp/nag_apple_cup.jpeg')
#img = cv2.imread('/Users/nravinuthala/temp/cutlery.jpeg')
img = cv2.imread('/Users/nravinuthala/temp/nag_bottle_laptop.jpeg')
img = cv2.resize(img, None, fx=0.5, fy=0.5)
height, width, channels = img.shape

def show(caption, img):
    cv2.imshow(caption, img)
    cv2.waitKey(0)
    cv2.destoyAllWindows()
#show("Image", img)
#Detect objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0,0,0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)
#print(outs[0][506][84])

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        #print(confidence)
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            #cv2.circle(img, (center_x, center_y), 10, (0, 0, 0), 2)

            #Rectangle coordinates
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_ITALIC
num_objects_detected = len(boxes)
for i in range(num_objects_detected):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        box_color = box_colors[i]
        label_color = label_colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(img, label, (int(x+w/2), y - 15), font, 0.5, label_color, 2)

show("Image", img)  

