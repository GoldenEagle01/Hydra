import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.createLBPHFaceRecognizer()
path = 'dataSet'

def getImagesWithID(path):
	nr = 0
	imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
	faces = []
	IDs = []
	for imagePath in imagePaths:
		if('jpg' in imagePath):
			faceImg = Image.open(imagePath).convert('L')
			faceNp = np.array(faceImg, 'uint8')
			ID = int(os.path.split(imagePath)[-1].split('.')[1])
			faces.append(faceNp)
			IDs.append(ID)
			nr = nr + 1 
			print nr
			cv2.imshow("training",faceNp)
			cv2.waitKey(10)
	return np.array(IDs), faces

IDs, faces = getImagesWithID(path)
recognizer.train(faces, IDs)
recognizer.save('trainingData.yml')
cv2.destroyAllWindows()


