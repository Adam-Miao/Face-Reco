import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer = cv2.createLBPHFaceRecognizer() # in OpenCV 2
recognizer.read('trainner/trainner.yml')
# recognizer.load('trainner/trainner.yml') # in OpenCV 2

cascade_path = r"cv2data\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
cam = cv2.VideoCapture(0)
# font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1) # in OpenCV 2
font = cv2.FONT_HERSHEY_SIMPLEX
z=True
while z:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
        img_id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        if conf > 61:
            if img_id == 1:
                print("Hi, welcome.")
                z = False
                break
        else:
            print(conf)
            img_id = "Unknown"
        # cv2.cv.PutText(cv2.cv.fromarray(im), str(Id), (x, y + h), font, 255)
        cv2.putText(im, str(img_id), (x, y + h), font, 0.55, (0, 255, 0), 1)
    cv2.imshow('im', im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
import cv2
import os
if not 'trainner' in os.listdir():
    try:
        os.mkdir()
    except:
        pass
import numpy as np
from PIL import Image

# recognizer = cv2.createLBPHFaceRecognizer()
detector = cv2.CascadeClassifier(r"cv2data\haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()


def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        image = Image.open(image_path).convert('L')
        image_np = np.array(image, 'uint8')
        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
            continue
        image_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(image_np)
        for (x, y, w, h) in faces:
            face_samples.append(image_np[y:y + h, x:x + w])
            ids.append(image_id)

    return face_samples, ids


faces, Ids = get_images_and_labels('dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')
os.remove("dataSet\\*.jpg")
