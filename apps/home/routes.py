# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
 
import json
import random
from flask_socketio import emit
from apps.home import blueprint
from flask import app, current_app, render_template, request,Response
from flask_login import login_required
from jinja2 import TemplateNotFound
import cv2
import face_recognition
import numpy as np
import os
from apps import db, login_manager
from datetime import datetime, time
from apps.authentication.models import Attendance, Employees, Visitors
from ..extensions import socketio

camera = cv2.VideoCapture(0)
# Load a sample picture and learn how to recognize it.
prasanth_image = face_recognition.load_image_file("/home/prasanth_km/liveprojects/facedetector/argon-dashboard-flask-master/apps/home/prasanth/emp.jpeg")
prasanth_face_encoding = face_recognition.face_encodings(prasanth_image)[0]

# Load a second sample picture and learn how to recognize it.
bradley_image = face_recognition.load_image_file("/home/prasanth_km/liveprojects/facedetector/argon-dashboard-flask-master/apps/home/bradley/emp.jpg")
bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    prasanth_face_encoding,
    bradley_face_encoding
]

known_face_names = [
    "emp001",
    "emp002"
]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames(app):  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

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
            
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                index +=1
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "visitor"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            
            print("face_names are")
            print(face_names)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4 

                print("face_names are")
                print(name)
                print("visitor" in name)
                if "visitor" in name:
                    semipath = "/static/assets/img/attendance/visitors/"+today
                    directory = "/home/prasanth_km/liveprojects/facedetector/argon-dashboard-flask-master/apps/"+semipath
                    imageName =name+str(random.randrange(1, 1000000))+"_"+ currenttime.replace(":", "_")+'_.jpg'

                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    if(len(os.listdir(directory)) <3):
                        face = frame[top:bottom, left:right] #slice the face from the image  
                        cv2.imwrite(os.path.join(directory ,imageName ), face) 
                        
                        with app.app_context():
                            visitor = Visitors(imageurl=semipath +"/"+ imageName,
                                            date=today,
                                            time=currenttime )
                            visitors.append(visitor.serialize())
                            db.session.add(visitor)
                            db.session.commit()

                    
                    
                else:
                    with app.app_context():
                        attendace = Attendance.query.filter_by(employeecode=name,date=today,attendancetype=attendanceType).first()
                        if attendace:
                            print ("Attendance Already Marked")
                        else:
                            semipath = "/static/assets/img/attendance/"+name+"/"+today
                            directory = "/home/prasanth_km/liveprojects/facedetector/argon-dashboard-flask-master/apps/"+semipath
                            if not os.path.exists(directory):
                                os.makedirs(directory)
                            employee = Employees.query.filter_by(employeecode=name).first()
                            imageName = attendanceType+"_"+currenttime.replace(":", "_")+'_.jpg'
                            if employee:
                                employees.append(employee.serialize(currenttime,semipath +"/"+ imageName))
                            face = frame[top:bottom, left:right] #slice the face from the image 
                            cv2.imwrite(directory +"/"+ imageName, face) 
                            attendace = Attendance(employeecode=name,date=today,attendancetype=attendanceType, time=currenttime,status="completed")
                            db.session.add(attendace)
                            db.session.commit()

                    semipath = "/static/assets/img/attendance/"+name+"/"+today
                    directory = "/home/prasanth_km/liveprojects/facedetector/argon-dashboard-flask-master/apps/"+semipath
                        
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    if(len(os.listdir(directory)) <10):
                        face = frame[top:bottom, left:right] #slice the face from the image 
                        imageName = currenttime.replace(":", "_")+'_.jpg'
                        cv2.imwrite(os.path.join(directory ,imageName ), face) 

                # Draw a box around the face a if a < b else b
                cv2.rectangle(frame, (left, top), (right, bottom),(0, 0, 255) if "visitor" in name else (0, 255, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255) if "visitor" in name else (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                


            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            socketio.emit("detected", { 
                                       "visitors":json.dumps(visitors),
                                       "employees":json.dumps(employees) } )
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@blueprint.route('/index')
@login_required
def index():

    db.session.query(Attendance).delete()
    db.session.commit()

    employee1 = Employees.query.filter_by(employeecode="emp001").first()
    
    if not employee1:
        employee1 = Employees(employeecode="emp001",
                              email="prasanth@gmail.com",
                              name="Prasanth K M", 
                              imageurl="/static/assets/img/employees/emp001.jpg",
                              designation="CTO")
        db.session.add(employee1)
        db.session.commit()
    
    employee2 = Employees.query.filter_by(employeecode="emp002").first()
    
    if not employee2:
        employee2 = Employees(employeecode="emp002",
                              email="bradley@gmail.com",
                              name="Bradley Kooper", 
                              imageurl="/static/assets/img/employees/emp002.jpg",
                              designation="CFO")
        db.session.add(employee2)
        db.session.commit()
 
    # , len = len(emps), employees=json.dumps(emps)
    return render_template('home/index.html', segment='index')

@blueprint.route('/profile')
@login_required
def profile():

    return render_template('home/profile.html', segment='profile')

@blueprint.route('/reports')
@login_required
def reports():

    return render_template('home/reports.html', segment='reports')
 
@blueprint.route('/video_feed')
def video_feed():
    app = current_app._get_current_object()
    return Response(gen_frames(app),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
