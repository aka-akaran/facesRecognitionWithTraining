import numpy as np
import cv2 as cv
import os



people = []
for i in os.listdir( r'U:\vscode\opencv\faces\train' ) :
    people.append(i)
# print(people)



haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')




img = cv.imread( r"U:\vscode\opencv\faces\train\Indira Gandhi\Indira Gandhi (4).jpg" )

gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY )
cv.imshow('Unknown', gray)




# # Detect the face in the image
faces = haar_cascade.detectMultiScale( gray, 1.1, 3 )

for( x,y,w,h ) in faces :
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print( f' Label = {people[label]} with a confidence of {confidence} ' )

    cv.putText( img, str(people[label]), (50, 150), cv.FONT_HERSHEY_COMPLEX,
                 1.0, (0,255,0), 2  )
    cv.rectangle( img, (x,y), (x+w, y+h), (255,0,0), 2 )

cv.imshow(f'Detected Face of {people[label]}', img)
cv.waitKey(0)