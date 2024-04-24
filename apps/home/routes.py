# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
 
import json
import random
from threading import Thread
from apps.home.camera import ThreadedCamera
from flask_socketio import emit
from apps.home import blueprint
from flask import app, current_app, render_template, request,Response
from flask_login import login_required
from jinja2 import TemplateNotFound
import cv2 
import numpy as np
import os
from apps import db
from apps.authentication.models import Attendance, Employees
  
 
@blueprint.route('/index')
@login_required
def index():

    db.session.query(Attendance).delete()
    db.session.query(Employees).delete()
    db.session.commit()

    employee1 = Employees.query.filter_by(employeecode="emp001").first()
    
    if not employee1:
        employee1 = Employees(employeecode="emp001",
                              email="ramzi@gmail.com",
                              name="Ramzi Rahman", 
                              imageurl="/static/assets/img/employees/ramzi.jpg",
                              designation="Project Engineer")
        db.session.add(employee1)
        db.session.commit()
    
    employee2 = Employees.query.filter_by(employeecode="emp002").first()
    
    if not employee2:
        employee2 = Employees(employeecode="emp002",
                              email="thasneem@gmail.com",
                              name="Thasneem Yousuf", 
                              imageurl="/static/assets/img/employees/thasneem.jpg",
                              designation="Accountant")
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

streamer = ThreadedCamera()

@blueprint.route('/video_feed')
def video_feed(): 
    streamer.restartThread()
    return Response(streamer.update(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@blueprint.route('/reset_data', methods=['GET'])
def reset_data():
    db.session.query(Attendance).delete()
    db.session.commit()
    return ("nothing")

     
    


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

 
    
# 5
# 16
# 159
# 120
# face_locations
# 44 5
# 112 16
# 95 159
# 60 129
# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
