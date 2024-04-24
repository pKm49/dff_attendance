from deepface import DeepFace
import glob
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load a model for Pose Estimation
pose_model = YOLO('yolov8n-pose.pt')

# Inputs - Please Enter the input Files paths here
cap = cv2.VideoCapture('/Users/kartikkhandelwal/Longspurtech/dataset_extras/1.mp4')
ramzi_pic = np.array(Image.open("/Users/kartikkhandelwal/Longspurtech/dataset/dataset/ramzi/frame560.jpg"))
thasneem_pic = np.array(Image.open("/Users/kartikkhandelwal/Longspurtech/dataset/dataset/thasneem/frame740.jpg"))

# get total number of frames and generate a list with each 30 th frame 
totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
x = [i for i in range (1, totalFrames) if divmod(i, int(100))[1]==0]

# print(type(ramzi_pic))

for myFrameNumber in x:
            #set which frame to read
            cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)
            # read frame
            ret, frame = cap.read()

            #### MY CODE HERE ####
            try:
                img = frame

                ## Pose Estimation ##           
                pose_results = pose_model(img)
                for r in pose_results[0]:
                    # if r.boxes.conf.cpu().numpy()[0] > 0.7:
                    print("*" * 20, r.boxes.conf.cpu().numpy()[0])
                    if r.boxes.conf.cpu().numpy()[0] > 0.8:
                        im_array = r.plot(labels=False, line_width=2)  # plot a BGR numpy array of predictions
                        ## Link for more parameters in r.plot() -> https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results.plot
                        frame = im_array  # RGB PIL image

                        x1,y1,x2,y2 = r.boxes.xyxy.cpu().numpy()[0].astype('int32')
                        person_cropped_image = im_array[y1:y2, x1:x2][:,:,::-1] ## <- Output 
                #####################

                ramzi = DeepFace.verify(ramzi_pic, img[:,:,::-1], model_name = "ArcFace", detector_backend = "retinaface")
                distance = str(round(ramzi["distance"], 2))
                if ramzi["verified"]:
                    x, y, w, h = ramzi["facial_areas"]["img2"]['x'], ramzi["facial_areas"]["img2"]['y'], ramzi["facial_areas"]["img2"]['w'], ramzi["facial_areas"]["img2"]['h']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Ramzi-' + distance, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    # cv2.imwrite(img_name, img)
                    person_crpped_face = img[y:y+h, x:x+w][:,:,::-1] ## <- Output
                
                thasneem = DeepFace.verify(thasneem_pic, img[:,:,::-1], model_name = "ArcFace", detector_backend = "retinaface")
                distance = str(round(thasneem["distance"], 2))
                if thasneem["verified"]:
                    x, y, w, h = thasneem["facial_areas"]["img2"]['x'], thasneem["facial_areas"]["img2"]['y'], thasneem["facial_areas"]["img2"]['w'], thasneem["facial_areas"]["img2"]['h']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Thasneem-' + distance, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    # cv2.imwrite(img_name, img)
                    person_crpped_face = img[y:y+h, x:x+w][:,:,::-1] ## <- Output

            except Exception as e: print(e); pass
            ######################

            # display frame
            cv2.imshow("video", frame)

            # wait one second, exit loop if escape is pressed
            ch = 0xFF & cv2.waitKey(100)
            if ch == 27:
                cv2.destroyAllWindows()
                break