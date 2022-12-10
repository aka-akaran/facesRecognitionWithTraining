import os
import cv2 as cv
import numpy as np


people = []
for i in os.listdir( r'U:\vscode\opencv\faces\train' ) :
    people.append(i)


DIR = r'U:\vscode\opencv\faces\train'
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


features = []
labels = []


def create_train() :
    for person in people :
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path) :
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor( img_array, cv.COLOR_BGR2GRAY )

            faces = face_cascade.detectMultiScale( gray, scaleFactor = 1.1,
            minNeighbors = 3 )

            for( x,y,w,z ) in faces :
                face_roi = gray[y:y+z, x:x+w]
                features.append(face_roi)
                labels.append(label)

create_train()

# print(f'Length of Features {len(features)}')
# print(f'Length of Labels {len(labels)}')
# print(people)


print('Training Complete --------------------- Training Complete')



features = np.array(features, dtype='object')
labels = np.array(labels)


face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on the features list and labels list
face_recognizer.train(features, labels )



# Save this training for using anywhere
face_recognizer.save('face_trained.yml')


np.save( 'features.npy', features )
np.save( 'labels.npy', labels )