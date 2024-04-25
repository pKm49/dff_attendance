# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
 
import json
import random
from threading import Thread
import traceback

from flask import current_app  
import cv2 
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
 
from PIL import Image

class ThreadedCamera():
    def __init__(self):
        self.app = current_app._get_current_object()
        self.parent_dir = "/home/prasanth_km/liveprojects/facedetector/argon-dashboard-flask-master/apps"
 
        self.video_db = self.parent_dir+"/static/assets/img/employees/"
        self.ramzi_db = self.parent_dir+"/static/assets/img/employees/ramzi/"
        self.thasneem_db = self.parent_dir+"/static/assets/img/employees/thasneem/"
        self.ramzi_pic = np.array(Image.open(self.ramzi_db+"emp.jpg"))
        self.thasneem_pic = np.array(Image.open(self.thasneem_db+"emp.jpg"))


        self.known_face_encodings = []
        self.known_face_names = []

         
        
        print("thasneem face_encoding completed") 

        self.capture = cv2.VideoCapture(self.video_db+"test.avi")
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
                      
                    ## Pose Estimation ##           
                    pose_results = pose_model(self.frame)
                    for r in pose_results[0]:
                        # if r.boxes.conf.cpu().numpy()[0] > 0.7:
                        # print(r.boxes)
                        # print("*" * 20, r.boxes.conf.cpu().numpy()[0])
                        if r.boxes.conf.cpu().numpy()[0] > 0.8: # RGB PIL image
                            pose_paths.append(r) 
        
                            # cv2.imwrite(os.path.join(directory ,imageName ), person_cropped_image) 

                    #####################        
                    # Only process every other frame of video to save time
                
                    # Find all the faces and face encodings in the current frame of video
                
                    employees = []
                    visitors = []
                    face_names = [] 
                    person_crpped_faces = [] 
                    person_crpped_face_coordinates = [] 
                     
                    today = datetime.now().strftime('%d-%m-%Y')
                    
                    attendanceType = "checkin" if datetime.now().time()<time(15,00) else "checkout"
                    currenttime = datetime.now().strftime('%H:%M:%S')
                    try:
                        semipath = "/static/assets/img/attendance/temp/"
                        directory = self.parent_dir+semipath
                        imageName =  "temp.jpg"

                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        cv2.imwrite(os.path.join(directory ,imageName ), self.frame) 

                        ramzi = DeepFace.verify(self.ramzi_pic, os.path.join(directory ,imageName ), model_name = "ArcFace", detector_backend = "retinaface")
                        thasneem = DeepFace.verify(self.thasneem_pic, os.path.join(directory ,imageName ), model_name = "ArcFace", detector_backend = "retinaface")
                        
                        # ramzi = DeepFace.verify(self.ramzi_pic, self.frame[:,:,::-1], model_name = "ArcFace", detector_backend = "retinaface")
                        print("ramzi is")
                        print(ramzi)
                        print(ramzi["distance"])
                        print(ramzi["distance"] > .9)
                        distance = str(round(ramzi["distance"], 2))

                        if ramzi["distance"]>thasneem["distance"]:
                            print("ramzi verified successfully")
                            x, y, w, h = ramzi["facial_areas"]["img2"]['x'], ramzi["facial_areas"]["img2"]['y'], ramzi["facial_areas"]["img2"]['w'], ramzi["facial_areas"]["img2"]['h']
                            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(self.frame, 'Ramzi', cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            # cv2.imwrite(img_name, img)
                            ramzi_crpped_face = self.frame[y:y+h, x:x+w][:,:,::-1] 
                            face_names.append("emp001")
                            person_crpped_faces.append(ramzi_crpped_face)
                            person_crpped_face_coordinates.append([x, y, x+w, y+h])
                        
                        else:
                            print("thasneem verified successfully")
                            x, y, w, h = thasneem["facial_areas"]["img2"]['x'], thasneem["facial_areas"]["img2"]['y'], thasneem["facial_areas"]["img2"]['w'], thasneem["facial_areas"]["img2"]['h']
                            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(self.frame, 'Thasneem', cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            # cv2.imwrite(img_name, img)
                            thasneem_crpped_face = self.frame[y:y+h, x:x+w][:,:,::-1] 
                            face_names.append("emp002")
                            person_crpped_faces.append(thasneem_crpped_face)
                            person_crpped_face_coordinates.append(thasneem_crpped_face)

                    except Exception as e:
                        print("face recognition error") 
                        pass
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
                    for person_crpped_face,person_crpped_face_coordinate, name in zip(person_crpped_faces,person_crpped_face_coordinates, face_names):
                        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                           
                        
                        if "visitor" in name:
                            semipath = "/static/assets/img/attendance/visitors/"+today
                            directory = self.parent_dir+semipath
                            imageName = name+str(random.randrange(1, 1000000))+"_"+ currenttime.replace(":", "_")+'_.jpg'

                            if not os.path.exists(directory):
                                os.makedirs(directory)

                            if(len(os.listdir(directory)) <3): 
                                cv2.imwrite(os.path.join(directory ,imageName ), person_crpped_face) 
                                
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
                                    imagePoseName =""  #slice the face from the image 
                                    if self.is_template_in_image(person_crpped_face_coordinate,pose_paths ):
                                        im_array = r.plot(labels=False)  # plot a BGR numpy array of predictions
                                        
                                        self.frame = im_array 
                                        x1,y1,x2,y2 = r.boxes.xyxy.cpu().numpy()[0].astype('int32')
                                        person_cropped_image = im_array[y1:y2, x1:x2]
                                        imagePoseName = attendanceType+"_pose_"+currenttime.replace(":", "_")+'_.jpg'
                                        cv2.imwrite(os.path.join(directory ,imagePoseName ), person_cropped_image) 
                                         
                                     
                                    if employee:
                                        employees.append(employee.serialize(currenttime,semipath +"/"+ imageName,semipath+"/"+imagePoseName) )
                                    

                                    cv2.imwrite(directory +"/"+ imageName, person_crpped_face) 
                                    attendace = Attendance(employeecode=name,date=today,attendancetype=attendanceType, time=currenttime,status="completed")
                                    db.session.add(attendace)
                                    db.session.commit()

                            semipath = "/static/assets/img/attendance/"+name+"/"+today
                            directory = self.parent_dir+semipath
    
                            if not os.path.exists(directory):
                                os.makedirs(directory)

                            if(len(os.listdir(directory)) <10):
                                imageName = currenttime.replace(":", "_")+'_.jpg'
                                cv2.imwrite(os.path.join(directory ,imageName ), person_crpped_face) 
 
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