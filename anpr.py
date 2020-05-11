import numpy as np
import argparse
import time
import cv2
import os
from PIL import Image
from datetime import datetime
import pytesseract
from xlwt import Workbook 
import os.path

#2,4,8 --Model Tuned on.
#1,3,5,6 --Need to tune.
image1="Image/2.jpg"
#excel sheet row column
row=1
column=1 
# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

# load YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load input image and grab its spatial dimensions
image = cv2.imread(image1)
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"] and classID in [2,3,5,7]:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
			now = datetime.now()
			today_date = now.strftime("%B %d, %Y")
			current_time = now.strftime("%H:%M:%S")
			current_time_string = now.strftime("%H-%M-%S-")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		#crop image and save
		Image.open(image1).crop((x, y, x+w, y+h)).save("Vehicle/"+current_time_string+"vehicle.jpg")

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

# show the output image
#cv2.imshow("Vehical", image)
#cv2.waitKey(0)

#code for number plate extraction starts from here


watch_cascade = cv2.CascadeClassifier('cascade.xml')
image = cv2.imread("Vehicle/"+current_time_string+"vehicle.jpg")
def detectPlateRough(image_gray,resize_h = 720,en_scale =1.08 ,top_bottom_padding_rate = 0.05):
        if top_bottom_padding_rate>0.2:
            print("error:top_bottom_padding_rate > 0.2:",top_bottom_padding_rate)
            exit(1)
        height = image_gray.shape[0]
        padding = int(height*top_bottom_padding_rate)
        scale = image_gray.shape[1]/float(image_gray.shape[0])
        image = cv2.resize(image_gray, (int(scale*resize_h), resize_h))
        image_color_cropped = image[padding:resize_h-padding,0:image_gray.shape[1]]
        image_gray = cv2.cvtColor(image_color_cropped,cv2.COLOR_RGB2GRAY)
        watches = watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9),maxSize=(36*40, 9*40))
        cropped_images = []
        for (x, y, w, h) in watches:

            #cv2.rectangle(image_color_cropped, (x, y), (x + w, y + h), (0, 0, 255), 1)
            #Tune this calculation, when image type and angle is fixed
            x -= w *0.004
            w += w * 0.5
            y -= h * 0.15
            h += h * 0.3

            #cv2.rectangle(image_color_cropped, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)

            cropped = cropImage(image_color_cropped, (int(x), int(y), int(w), int(h)))
            cropped_images.append([cropped,[x, y+padding, w, h]])
            
#Remove comments to see the cropped images            
def cropImage(image,rect):
        #cv2.imshow("Orignal Image", image)
        #cv2.waitKey(0)
        x, y, w, h = computeSafeRegion(image.shape,rect)
        #cv2.imshow("NumberPlate",image[y:y+h,x:x+w] )
        cv2.imwrite("Number_plate/"+current_time_string+'numberplate.jpg', image[y:y+h,x:x+w])
        #cv2.waitKey(0)
        return image[y:y+h,x:x+w]


def computeSafeRegion(shape,bounding_rect):
        top = bounding_rect[1] # y
        bottom  = bounding_rect[1] + bounding_rect[3] # y +  h
        left = bounding_rect[0] # x
        right =   bounding_rect[0] + bounding_rect[2] # x +  w
        min_top = 0
        max_bottom = shape[0]
        min_left = 0
        max_right = shape[1]

        #print(left,top,right,bottom)
        #print(max_bottom,max_right)

        if top < min_top:
            top = min_top
        if left < min_left:
            left = min_left
        if bottom > max_bottom:
            bottom = max_bottom
        if right > max_right:
            right = max_right
        return [left,top,right-left,bottom-top]

detectPlateRough(image,image.shape[0],top_bottom_padding_rate=0.1)

wb = Workbook()
sheet1 = wb.add_sheet(today_date) 
sheet1.write(0, 0,"Time")
sheet1.write(0,1,"Vehical Image_Name")
sheet1.write(0, 2, 'Car_Number') 
sheet1.write(row, column-1, current_time) 
sheet1.write(row,column,current_time_string+"vehicle.jpg")
# code for ocr starts here.

if(os.path.isfile("Number_plate/"+current_time_string+'numberplate.jpg')!=True):
    sheet1.write(row, column+1, "NumberNotDetected")
    row=row+1
    wb.save("Entry Record"+".xls")
else:
    img = cv2.imread("Number_plate/"+current_time_string+'numberplate.jpg')
    #Resize
    width = 157
    height = 29
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    #grayscale
    img = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    
    #noise reduction
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=3)
    img = cv2.erode(img, kernel, iterations=3)
    
    #blurring
    img = cv2.GaussianBlur(img, (1, 1), 0)
    img = cv2.medianBlur(img, 3)
    
    #binarize
    cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, img)
    
    text = pytesseract.image_to_string(Image.open(filename), lang='eng',
            config='--psm 7 ')
    os.remove(filename)
    print("Number is : "+text)
    sheet1.write(row, column+1, text)
    row=row+1
    wb.save("Excel_sheet/"+"Entry Record"+".xls")
