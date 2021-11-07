import cv2
import numpy as np
import time
import imutils

net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") #Tiny Yolo
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

def issit(h):
    if float(h) <= 500:  #change as per ur needs!!!!!
        return True
    else:
        return False


#loading image
cap=cv2.VideoCapture("test_video.mp4") #0 for 1st webcam
 
font = cv2.FONT_HERSHEY_PLAIN
starting_time= time.time()
frame_id = 0
obj_id = 0
    
while True:
    _,frame= cap.read() # 
    frame_id+=1
    
    height,width,channels = frame.shape
    #detecting objects
    blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False) #reduce 416 to 320    

        
    net.setInput(blob)
    outs = net.forward(outputlayers)
    #print(outs[1])


    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]
    TrackedIDs = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4 and class_id == 0:
                #onject detected
                
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #get ID
                Id = int(obj_id)                
                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence= confidences[i]
            font = cv2.FONT_HERSHEY_DUPLEX
############################################################################################
            #print(h)
            if issit(h):
                cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 0, 255), 2)
                cv2.putText(frame, 'SITTING', (x,y-10), font, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(frame, 'STANDING', (x,y-10), font, 0.5, (0, 255, 255), 1)
                cv2.rectangle(frame, (x,y),(x+w,y+h), (0, 255, 0), 2)

            

    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,1,(255,255,0),1)
    frame = imutils.resize(frame, width = 800)
    cv2.imshow("Image",frame)
    key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
    
    if key == 27: #esc key stops the process
        break;
    
cap.release()    
cv2.destroyAllWindows()
