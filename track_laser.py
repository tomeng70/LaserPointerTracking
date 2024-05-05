import cv2
import numpy as np
cap = cv2.VideoCapture(0)

CIRCLE_DIA = 20
CIRCLE_RAD = 2
CIRCLE_COLOR = (0, 0, 255)      # rgb
BAR_HEIGHT = 28
CIRCLE_OFFSET = BAR_HEIGHT // 2 - 4

prevState = 0
currState = 0
pts = []
while (1):

    # Take each frame
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #lower_red = np.array([170, 50, 50])
    lower_red = np.array([160, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)

    if (minVal > 0 or maxVal > 0):
        #print(f"minVal = {minVal:06.2f}, maxVal = {maxVal:06.2f}, " 
        #    + f"minLoc = {minLoc}, maxLoc = {maxLoc}")
        cp = np.add(maxLoc, (0, CIRCLE_OFFSET))
        # only add the first point detected in a burst.
        if (prevState == 0):
            pts.append(cp)
            prevState = 1
        ##cv2.circle(frame, cp, CIRCLE_DIA, CIRCLE_COLOR, CIRCLE_RAD, cv2.LINE_AA)
    else:
        if (prevState == 1):
            prevState = 0
    # loop through pts.
    for pt in pts:
        cv2.circle(frame, pt, CIRCLE_DIA, CIRCLE_COLOR, CIRCLE_RAD, cv2.LINE_AA)
    
    cv2.imshow('Track Laser', frame)
    #cv2.imshow('Track Laser', hsv)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
