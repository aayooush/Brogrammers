from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow as tf
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# Define a flask app
app = Flask(__name__)

print('Check http://127.0.0.1:5000/')



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get the file from post request
        u = request.form['username']
        p = request.form['password']
        
        print(u,p)
        cnt = [1,1,1,1,1,1]
        
        if p == 'a':
            return render_template('landing.html',username=u,passw=p,count=cnt)
        else:
            return render_template('login.html')        
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
