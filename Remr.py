"""
The script to learn faces
"""
import cv2
import os
if not 'dataSet' in os.listdir():
    try:
        os.mkdir()
    except:
        pass
import numpy
print("Scanning setector")
detector = cv2.CascadeClassifier(r'cv2data\haarcascade_frontalface_default.xml')
print("Capturing video")
cap = cv2.VideoCapture(0)
print("Done.")

sampleNum = 0
Id = input('Enter your id: ')

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # incrementing sample number
        sampleNum = sampleNum + 1
        # saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User." + str(Id) + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])  #

        cv2.imshow('frame', img)
    # wait for 40 miliseconds
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 255
    elif sampleNum > 255:
        break

cap.release()
cv2.destroyAllWindows()
