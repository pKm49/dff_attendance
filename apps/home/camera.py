# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
 
import json
import random
from threading import Thread

from flask import current_app  
import cv2
import face_recognition
import numpy as np
import os
from apps import db
from datetime import datetime, time
from apps.authentication.models import Attendance, Employees, Visitors
from ..extensions import socketio

import cv2
from threading import Thread
from ultralytics import YOLO
pose_model = YOLO('yolov8n-pose.pt')
from deepface import DeepFace

import face_recognition
from PIL import Image

class ThreadedCamera():
    def __init__(self):
        self.app = current_app._get_current_object()
        self.parent_dir = "/home/prasanth_km/liveprojects/facedetector/argon-dashboard-flask-master/apps"
 
        self.ramzi_db = self.parent_dir+"/static/assets/img/employees/ramzi/"
        self.thasneem_db = self.parent_dir+"/static/assets/img/employees/thasneem/"
        self.ramzi_pic = np.array(Image.open(self.ramzi_db+"emp.jpg"))


        self.known_face_encodings = []
        self.known_face_names = []

        # for file in os.listdir(self.ramzi_db):
        #     filename = os.fsdecode(file)  
        #     ramzi_image = face_recognition.load_image_file(os.path.join(self.ramzi_db, filename))
        #     face_encoding = face_recognition.face_encodings(ramzi_image)
        #     print("ramzi face_encoding") 
        #     if len(face_encoding)>0:
        #         self.known_face_encodings.append(face_recognition.face_encodings(ramzi_image)[0])
        #         self.known_face_names.append("emp001")
        
        # print("ramzi face_encoding completed") 

        # for file in os.listdir(self.thasneem_db):
        #     filename = os.fsdecode(file)  
        #     thasneem_image = face_recognition.load_image_file(os.path.join(self.thasneem_db, filename))
        #     face_encoding = face_recognition.face_encodings(thasneem_image)
        #     print("thasneem face_encoding") 
        #     if len(face_encoding)>0:
        #         self.known_face_encodings.append(face_recognition.face_encodings(thasneem_image)[0])
        #         self.known_face_names.append("emp002")
        
        print("thasneem face_encoding completed") 

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
       
        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)
        
        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame  = None

    def restartThread(self): 
        self.thread = None 
        
        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True: 
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()   # read the camera frame
                if not self.status:
                    break
                else:
                    pose_paths = [] 
                    # Resize frame of video to 1/4 size for faster face recognition processing
                    small_frame = cv2.resize(self.frame, (0, 0), fx=0.25, fy=0.25)
                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    rgb_small_frame = small_frame[:, :, ::-1]

                    ## Pose Estimation ##           
                    pose_results = pose_model(rgb_small_frame)
                    for r in pose_results[0]:
                        # if r.boxes.conf.cpu().numpy()[0] > 0.7:
                        # print(r.boxes)
                        # print("*" * 20, r.boxes.conf.cpu().numpy()[0])
                        if r.boxes.conf.cpu().numpy()[0] > 0.8:
                            
                            pose_paths.append( r)
        
                            # cv2.imwrite(os.path.join(directory ,imageName ), person_cropped_image) 

                    #####################        
                    # Only process every other frame of video to save time
                
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    employees = []
                    visitors = []
                    face_names = [] 
                    
                    index = 0
                    today = datetime.now().strftime('%d-%m-%Y')
                    
                    attendanceType = "checkin" if datetime.now().time()<time(15,00) else "checkout"
                    currenttime = datetime.now().strftime('%H:%M:%S')
                    try:
                        ramzi = DeepFace.verify(self.ramzi_pic, self.frame[:,:,::-1], model_name = "ArcFace", detector_backend = "retinaface")
                        distance = str(round(ramzi["distance"], 2))
                        if ramzi["verified"]:
                            print("ramzi verified successfully")
                            x, y, w, h = ramzi["facial_areas"]["img2"]['x'], ramzi["facial_areas"]["img2"]['y'], ramzi["facial_areas"]["img2"]['w'], ramzi["facial_areas"]["img2"]['h']
                            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(self.frame, 'Ramzi-' + distance, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            # cv2.imwrite(img_name, img)
                            person_crpped_face = self.frame[y:y+h, x:x+w][:,:,::-1]
                            semipath = "/static/assets/img/attendance/visitors/"+today
                            directory = self.parent_dir+semipath
                            imageName = name+str(random.randrange(1, 1000000))+"_"+ currenttime.replace(":", "_")+'_.jpg'
                            if not os.path.exists(directory):
                                os.makedirs(directory)
                            cv2.imwrite(os.path.join(directory ,imageName ), person_crpped_face)
                                
                    except Exception as e: print(e); pass
                    # for face_encoding in face_encodings:
                    #     # See if the face is a match for the known face(s)
                    #     index +=1
                    #     matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    #     name = "visitor"
                    #     # Or instead, use the known face with the smallest distance to the new face
                    #     face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                         
                    #     best_match_index = np.argmin(face_distances)
                    #     if matches[best_match_index]:
                    #         name = self.known_face_names[best_match_index]

                    #     face_names.append(name)
                    

                    # Display the results
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                         
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4 
                        
                        
                        if "visitor" in name:
                            semipath = "/static/assets/img/attendance/visitors/"+today
                            directory = self.parent_dir+semipath
                            imageName = name+str(random.randrange(1, 1000000))+"_"+ currenttime.replace(":", "_")+'_.jpg'

                            if not os.path.exists(directory):
                                os.makedirs(directory)

                            if(len(os.listdir(directory)) <3):
                                face = self.frame[top:bottom, left:right] #slice the face from the image  
                                cv2.imwrite(os.path.join(directory ,imageName ), face) 
                                
                                with self.app.app_context():
                                    visitor = Visitors(imageurl=semipath +"/"+ imageName,
                                                    date=today,
                                                    time=currenttime )
                                    visitors.append(visitor.serialize())
                                    db.session.add(visitor)
                                    db.session.commit()
        
                        else:
                            with self.app.app_context():
                                attendace = Attendance.query.filter_by(employeecode=name,date=today,attendancetype=attendanceType).first()
                                if attendace:
                                    print ("Attendance Already Marked")
                                else:
                                    semipath = "/static/assets/img/attendance/"+name+"/"+today
                                    directory = self.parent_dir+semipath
                                    if not os.path.exists(directory):
                                        os.makedirs(directory)
                                    employee = Employees.query.filter_by(employeecode=name).first()
                                    imageName = attendanceType+"_"+currenttime.replace(":", "_")+'_.jpg'
                                    imagePoseName =""
                                    face = self.frame[top:bottom, left:right] #slice the face from the image 
                                    if self.is_template_in_image(face,pose_paths ):
                                        im_array = r.plot(labels=False)  # plot a BGR numpy array of predictions
                                        ## Link for more parameters in r.plot() -> https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results.plot
                                        # frame = im_array  # RGB PIL image
                    
                                        x1,y1,x2,y2 = r.boxes.xyxy.cpu().numpy()[0].astype('int32')
                                        person_cropped_image = im_array[y1:y2, x1:x2]
                                        imagePoseName = attendanceType+"_pose_"+currenttime.replace(":", "_")+'_.jpg'
                                        cv2.imwrite(os.path.join(directory ,imagePoseName ), person_cropped_image) 
                                         
                                     
                                    if employee:
                                        employees.append(employee.serialize(currenttime,semipath +"/"+ imageName,semipath+"/"+imagePoseName) )
                                    

                                    cv2.imwrite(directory +"/"+ imageName, face) 
                                    attendace = Attendance(employeecode=name,date=today,attendancetype=attendanceType, time=currenttime,status="completed")
                                    db.session.add(attendace)
                                    db.session.commit()

                            semipath = "/static/assets/img/attendance/"+name+"/"+today
                            directory = self.parent_dir+semipath
    
                            if not os.path.exists(directory):
                                os.makedirs(directory)

                            if(len(os.listdir(directory)) <10):
                                face = self.frame[top:bottom, left:right] #slice the face from the image 
                                imageName = currenttime.replace(":", "_")+'_.jpg'
                                cv2.imwrite(os.path.join(directory ,imageName ), face) 

                        # Draw a box around the face a if a < b else b
                        cv2.rectangle(self.frame, (left, top), (right, bottom),(0, 0, 255) if "visitor" in name else (0, 255, 0), 2)

                        # Draw a label with a name below the face
                        cv2.rectangle(self.frame, (left, bottom - 35), (right, bottom), (0, 0, 255) if "visitor" in name else (0, 255, 0), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(self.frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                        

                    ret, buffer = cv2.imencode('.jpg', self.frame)
                    self.frame = buffer.tobytes()
                
                    socketio.emit("detected", { 
                                            "visitors":json.dumps(visitors),
                                            "employees":json.dumps(employees) } )
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + self.frame + b'\r\n') 
                    

    def is_template_in_image(self,img, templList):
        for templ in templList:
            x1,y1,x2,y2 = templ.boxes.xyxy.cpu().numpy()[0].astype('int32')
            in_range_along_x = ((img[0] > x1*4) & (img[1] > y1*4)).all()
            in_range_along_y = ((img[2] < x2*4) & (img[3] < y2*4)).all()
            if in_range_along_x and in_range_along_y:
                return True
        
        return False

    def grab_frame(self):
        print("self.status")
        print(self.status)
        if self.status:
            return self.frame
        return None  