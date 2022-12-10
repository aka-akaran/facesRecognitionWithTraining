import cv2 as cv


img = cv.imread('Media/Photos/7.jpg')
cv.imshow('Original', img)

gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY )
cv.imshow('Gray Image', gray)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale( gray, scaleFactor = 1.1, minNeighbors = 7 )
print( f'Number of faces = {len(faces)}' )

for( x, y, w, z ) in faces :
    cv.rectangle( img, (x,y), (x+w, y+x), (0,0,255), 2 )

cv.imshow( 'Faces', img )










cv.waitKey(0)
cv.destroyAllWindows