from utils import CFEVideoConf, image_resize
import glob
import math
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

framesPerSecond = 24.0
savePath = 'filter.mp4'
config = CFEVideoConf(cap, filepath=savePath, res='480p')

# Filters
def applyInvert(frame):
	return cv2.bitwise_not(frame)


def verifyAlphaChannel(frame):
	try:
		frame.shape[3]
	except IndexError:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
	return frame


def applyColor(frame, intensity=0.2, blue=0, green=0, red=0):
	
	frame = verifyAlphaChannel(frame)
	frameH, frameW, frameC = frame.shape
	sepiaBGRA = (blue, green, red, 1)
	overlay = np.full((frameH, frameW, 4), sepiaBGRA, dtype='uint8')
	cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
	return frame

def applySepia(frame, intensity=1.0, blue=20, green=66, red=112):
	return applyColor(frame, intensity, blue=blue, green=green, red=red)

def applyReddish(frame, intensity=0.2, blue=10, red=240):
	return applyColor(frame, intensity=intensity, blue=blue, red=red)

def applyGreenish(frame, intensity=0.2, blue=10, green=240):
	return applyColor(frame, intensity=intensity, blue=blue, green=green)

def applyBlueish(frame, intensity=0.2, blue=240, green=10):
	return applyColor(frame, intensity=intensity, blue=blue, green=green)

def alphaBlend(frame1, frame2, mask):
	alpha = mask/255.0
	blended = cv2.convertScaleAbs(frame1*(1-alpha) + frame2*alpha)
	return blended

def applyCenterFocus(frame, intensity=0.2):
	frame = verifyAlphaChannel(frame)
	frameH, frameW, frameC = frame.shape
	y = int(frameH/2)
	x = int(frameW/2)
	radius = int(y/2)
	center = (x,y)

	mask = np.zeros((frameH, frameW, 4), dtype='uint8')
	cv2.circle(mask, center, radius, (255, 255, 255), -1, cv2.LINE_AA)

	mask = cv2.GaussianBlur(mask, (21,21), 11)
	blured = cv2.GaussianBlur(frame, (21,21), 11)

	blended = alphaBlend(frame, blured, 255-mask)
	frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)

	return frame






while True:

	ret, frame = cap.read()
	intensity = 0.2
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow('Blue frame', gray)

	# blueFilter = applyBlueish(frame.copy(), intensity)
	# cv2.imshow('Blue frame', blueFilter)

	# greenFilter = applyGreenish(frame.copy(), intensity)
	# cv2.imshow('Green frame', greenFilter)

	# redFilter = applyReddish(frame.copy(), intensity)
	# cv2.imshow('Red frame', redFilter)

	# sepiaFilter = applySepia(frame.copy(), intensity)
	# cv2.imshow('Sepia frame', sepiaFilter)

	circleFocusFilter = applyCenterFocus(frame.copy(), intensity)
	cv2.imshow('Circle Focus frame', circleFocusFilter)

	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()