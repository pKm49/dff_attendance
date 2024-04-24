# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask_login import UserMixin

from apps import db, login_manager

from apps.authentication.util import hash_pass

class Users(db.Model, UserMixin):

    __tablename__ = 'Users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True)
    email = db.Column(db.String(64), unique=True)
    password = db.Column(db.LargeBinary)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            # depending on whether value is an iterable or not, we must
            # unpack it's value (when **kwargs is request.form, some values
            # will be a 1-element list)
            if hasattr(value, '__iter__') and not isinstance(value, str):
                # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
                value = value[0]

            if property == 'password':
                value = hash_pass(value)  # we need bytes here (not plain str)

            setattr(self, property, value)

    def __repr__(self):
        return str(self.username)
 

 
class Employees(db.Model):

    __tablename__ = 'Employees'

    id = db.Column(db.Integer, primary_key=True)
    employeecode = db.Column(db.String(64), unique=True)
    imageurl = db.Column(db.String(64), unique=True)
    email = db.Column(db.String(64), unique=True) 
    name = db.Column(db.String(64), unique=False) 
    designation = db.Column(db.String(64), unique=False) 

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            # depending on whether value is an iterable or not, we must
            # unpack it's value (when **kwargs is request.form, some values
            # will be a 1-element list)
            if hasattr(value, '__iter__') and not isinstance(value, str):
                # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
                value = value[0] 

            setattr(self, property, value)
 
    def serialize(self,time,attendanceimageurl,poseimageurl):
       """Return object data in easily serializable format"""
       return {
           'id'         : self.id,
           'employeecode': self.employeecode, 
           'time': time, 
           'email': self.email, 
           'imageurl': self.imageurl, 
           'attendanceimageurl': attendanceimageurl, 
           'poseimageurl': poseimageurl, 
           'name': self.name, 
           'designation': self.designation, 
       }
 
class Attendance(db.Model):

    __tablename__ = 'Attendance'

    id = db.Column(db.Integer, primary_key=True)
    employeecode = db.Column(db.String(64), unique=True)
    date = db.Column(db.String(64), unique=False) 
    attendancetype = db.Column(db.String(64), unique=False)  
    time = db.Column(db.String(64), unique=False)  
    status = db.Column(db.String(64), unique=False)  

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            # depending on whether value is an iterable or not, we must
            # unpack it's value (when **kwargs is request.form, some values
            # will be a 1-element list)
            if hasattr(value, '__iter__') and not isinstance(value, str):
                # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
                value = value[0] 

            setattr(self, property, value)
 
class Visitors(db.Model):

    __tablename__ = 'Visitors'

    id = db.Column(db.Integer, primary_key=True) 
    date = db.Column(db.String(64), unique=False)  
    time = db.Column(db.String(64), unique=False)  
    imageurl = db.Column(db.String(64), unique=False)  

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            # depending on whether value is an iterable or not, we must
            # unpack it's value (when **kwargs is request.form, some values
            # will be a 1-element list)
            if hasattr(value, '__iter__') and not isinstance(value, str):
                # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
                value = value[0] 

            setattr(self, property, value)

    def serialize(self):
       """Return object data in easily serializable format"""
       return {
           'id'         : self.id,  
           'time': self.time, 
           'date': self.date, 
           'imageurl': self.imageurl 
       }
 
 

@login_manager.user_loader
def user_loader(id):
    return Users.query.filter_by(id=id).first()


@login_manager.request_loader
def request_loader(request):
    username = request.form.get('username')
    user = Users.query.filter_by(username=username).first()
    return user if user else None
