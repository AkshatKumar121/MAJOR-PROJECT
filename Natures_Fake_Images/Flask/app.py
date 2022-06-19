from flask import Flask, render_template, request, redirect, url_for,session
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2 as cv
import pandas as pd
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'fakeimage'

dic = {0: 'Fake_image', 1: 'Real_image'}
f = Path("models/CNN_model_structure.json")
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("models/CNN_model_weights.h5")

f1 = Path("models/gab_model_structure.json")
model_structure1 = f1.read_text()
model1 = model_from_json(model_structure1)
model1.load_weights("models/gab_model_weights.h5")


def predict_label(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    p = model.predict(x)
    x = np.argmax(p)
    return dic[x]


def predict_label1(img):
    img = cv.imread(img,0)
    gabor_1 = cv.getGaborKernel((18, 18), 1.5, np.pi/4, 5.0, 1.5, 0, ktype=cv.CV_32F) #initialising the parameters of gabor filter 
    filtered_img_1 = cv.filter2D(img, cv.CV_8UC3, gabor_1) # applying gabor filter
    gabor_2 = cv.getGaborKernel((18, 18), 1.5, np.pi/4, 5.0, 1.5, 0, ktype=cv.CV_32F)
    filtered_img_2 = cv.filter2D(filtered_img_1, cv.CV_8UC3, gabor_2)
    cv.imwrite('static/input/test.jpg', filtered_img_2)
    img = image.load_img("static/input/test.jpg", target_size=(128, 128))
    # img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    p = model1.predict(x)
    x = np.argmax(p)
    return dic[x]

# routes

@app.route('/')
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form["email"]
        pwd = request.form["password"]
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows():
            if row["email"] == str(email) and row["password"] == str(pwd):
                return redirect(url_for('home'))
        else:
            mesg = 'Invalid Login Try Again'
            return render_template('login.html', msg=mesg)
    return render_template('login.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['Email']
        password = request.form['Password']
        col_list = ["name", "email", "password"]
        r1 = pd.read_excel('user.xlsx', usecols=col_list)
        new_row = {'name': name, 'email': email, 'password': password}
        r1 = r1.append(new_row, ignore_index=True)
        r1.to_excel('user.xlsx', index=False)
        print("Records created successfully")
        # msg = 'Entered Mail ID Already Existed'
        msg = 'Registration Successful !! U Can login Here !!!'
        return render_template('login.html', msg=msg)
    return render_template('register.html')


@app.route("/home", methods=['GET', 'POST'])
def home():
   return render_template("home.html")

@app.route("/face", methods=['GET', 'POST'])
def face():
   return render_template("face.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_hours():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
        return render_template("home.html", prediction=p, img_path=img_path)


@app.route("/submit1", methods=['GET', 'POST'])
def get_hours1():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict_label1(img_path)
        return render_template("face.html", prediction=p, img_path=img_path)


@app.route('/password', methods=['POST', 'GET'])
def password():
    if request.method == 'POST':
        current_pass = request.form['current']
        new_pass = request.form['new']
        verify_pass = request.form['verify']
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows():
            if row["password"] == str(current_pass):
                if new_pass == verify_pass:
                    r1.replace(to_replace=current_pass, value=verify_pass, inplace=True)
                    r1.to_excel("user.xlsx", index=False)
                    msg1 = 'Password changed successfully'
                    return render_template('password_change.html', msg1=msg1)
                else:
                    msg2 = 'Re-entered password is not matched'
                    return render_template('password_change.html', msg2=msg2)
        else:
            msg3 = 'Incorrect password'
            return render_template('password_change.html', msg3=msg3)
    return render_template('password_change.html')


@app.route('/graphs', methods=['POST', 'GET'])
def graphs():
    return render_template('graphs.html')


@app.route('/cnn')
def cnn():
    return render_template('cnn.html')


@app.route('/logout')
def logout():
    session.clear()
    msg='You are now logged out', 'success'
    return redirect(url_for('login', msg=msg))


if __name__ == '__main__':
    app.run(debug=True)
