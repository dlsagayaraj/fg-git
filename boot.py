import cv2,os
import sys
import numpy as np
from PIL import Image
from Face import get_images_and_labels
import Tkinter as tk
import cv2
from PIL import Image, ImageTk

w, h = 600, 400




cascPath = "./config/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
video_capture.set(3,600)
video_capture.set(4,400)
# Path to the images
path = './images'
# Call the get_images_and_labels function and get the face images and the
# corresponding labels
images, labels = get_images_and_labels(path)
print len(labels)
# Perform the tranining

# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.createLBPHFaceRecognizer()
recognizer.train(images, np.array(labels))
people={'8853':"Daniel",'8046':"Clement",'8446':"Naveen",'8734':"Abdhulla",'8651':"Sridhar",'8822':"Venkat"}
print people




root = tk.Tk()
# get screen width and height
ws = root.winfo_screenwidth() # width of the screen
hs = root.winfo_screenheight() # height of the screen

# calculate x and y coordinates for the Tk root window
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)

# set the dimensions of the screen
# and where it is placed
root.geometry('%dx%d+%d+%d' % (w, h, x, y))
root.overrideredirect(1)
root.bind('<Escape>', lambda e: root.quit())
lmain = tk.Label(root)
lmain.pack()

def show_frame():
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    predict_image = np.array(gray, 'uint8')
    faces = faceCascade.detectMultiScale(
        predict_image,
        scaleFactor=1.9,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        print nbr_predicted,conf
        #distance = 1.0f - sqrt( distSq / (float)(nTrainFaces * nEigens) ) / 255.0f
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,people[str(nbr_predicted)],((w/2)-10,(h/2)-10), font, 1,(255,255,255),2,3)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()
root.mainloop()

