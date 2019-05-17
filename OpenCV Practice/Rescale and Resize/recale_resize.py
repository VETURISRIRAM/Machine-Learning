import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def make1080p():
	cap.set(3, 1920)
	cap.set(4, 1080)

def make720p():
	cap.set(3, 1280)
	cap.set(4, 720)

def make480p():
	cap.set(3, 640)
	cap.set(4, 480)

def changeResolution(width, height):
	cap.set(3, width)
	cap.set(4, height)

def rescaleFrame(frame, percent=75):
	width = int(frame.shape[1] * percent / 100)
	height = int(frame.shape[0] * percent / 100)
	dim = (width, height)
	return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)





while True:
	ret, frame = cap.read()
	frame = rescaleFrame(frame, percent=100)
	cv2.imshow('frame', frame)
	
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()