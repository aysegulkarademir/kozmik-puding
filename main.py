import cv2

face_detec = cv2.CascadeClassifier(r"classifier/haarcascade_frontalface_default.xml")

img = cv2.imread(r"photos/0.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

faces = face_detec.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x+w, y+h), (0,255,0),3)

cv2.imshow('img', img)
cv2.waitKey(0)