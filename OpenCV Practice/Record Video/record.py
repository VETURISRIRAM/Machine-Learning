import os
import numpy as np
import cv2

filename = 'video.mp4'
framesPerSecond = 24.0
resolution = '720p'

def changeResolution(cap, width, height):
	cap.set(3, width)
	cap.set(4, height)

# Standard Dimensions
STD_DIMS = {
	'480p' : (640, 480),
	'720p' : (1280, 720),
	'1080p' : (1960, 1080),
	'4k' : (3840, 2160)
}

def getDims(cap, res='1080p'):
	width, height = STD_DIMS['480p']
	if res in STD_DIMS:
		width, height = STD_DIMS[res]
	changeResolution(cap, width, height)
	return width, height

VIDEO_TYPE = {
	'avi' : cv2.VideoWriter_fourcc(*'XVID'),
	'mp4' : cv2.VideoWriter_fourcc(*'XVID')
}

def getVideoType(filename):
	filename, ext = os.path.splitext(filename)
	if ext in VIDEO_TYPE:
		return VIDEO_TYPE[ext]
	return VIDEO_TYPE['avi']

cap = cv2.VideoCapture(0)
dims = getDims(cap, res=resolution)
videoTypeCV2 = getVideoType(filename)

out = cv2.VideoWriter(filename, videoTypeCV2, framesPerSecond, dims)

while True:
	ret, frame = cap.read()
	out.write(frame)
	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllWindows()