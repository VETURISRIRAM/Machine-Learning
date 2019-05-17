import numpy as np
import cv2


faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')


cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)
	for (x, y, w, h) in faces:
		print(x, y, w, h)
		regionOfInterestGray = gray[y:y+h, x:x+w]
		regionOfInterestColor = gray[y:y+h, x:x+w]
		imageName = 'face.png'
		cv2.imwrite(imageName, regionOfInterestGray)

		color = (255, 0, 0) # BGR  
		stroke = 2
		endCoordX = x + w
		endCoordY = y + h
		cv2.rectangle(frame, (x, y), (endCoordX, endCoordY), color, stroke)

	cv2.imshow('frame', frame)
	
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()