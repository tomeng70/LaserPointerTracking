import cv2
import numpy as np


CIRCLE_DIA = 20
CIRCLE_RAD = 2
CIRCLE_COLOR = (0, 0, 255)      # rgb
BAR_HEIGHT = 28
CIRCLE_OFFSET = BAR_HEIGHT // 2 - 4

MIN_HUE = 20
MAX_HUE = 160
MIN_SAT = 100
MAX_SAT = 255
MIN_VAL = 200
MAX_VAL = 256

# global variables
cap = cv2.VideoCapture(0)
prevState = 0
pts = []
min_hue = MIN_HUE
max_hue = MAX_HUE
min_sat = MIN_SAT
max_sat = MAX_SAT
min_val = MIN_VAL
max_val = MAX_VAL


def findCenter(frame):
    center = None
    radius = None
    countours = cv2.findContours(frame, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]

    # only proceed if at least one contour was found
    if len(countours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(countours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        moments = cv2.moments(c)
        if moments["m00"] > 0:
            center = int(moments["m10"] / moments["m00"]), \
                     int(moments["m01"] / moments["m00"])
        else:
                center = int(x), int(y)

    # return center
    return center

def main():
    global prevState
    global pts
    global cap
    while (1):
        # Take each frame
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # split into hue, sat, and val components.
        h, s, v = cv2.split(hsv)

        # process hue
        (t, tmp) = cv2.threshold(h, max_hue, 0, cv2.THRESH_TOZERO_INV)
        (t, h) = cv2.threshold(tmp, min_hue, 255, cv2.THRESH_BINARY)
        h = cv2.bitwise_not(h)

        # process sat
        (t, tmp) = cv2.threshold(s, max_sat, 0, cv2.THRESH_TOZERO_INV)
        (t, s) = cv2.threshold(tmp, min_sat, 255, cv2.THRESH_BINARY)

        # process val
        (t, tmp) = cv2.threshold(v, max_val, 0, cv2.THRESH_TOZERO_INV)
        (t, v) = cv2.threshold(tmp, min_val, 255, cv2.THRESH_BINARY)

        # recombine values again.
        laser = cv2.bitwise_and(h, v)
        laser = cv2.bitwise_and(s, laser)

        merged = cv2.merge([h, s, v])

        # find contours to locate laser.
        center = findCenter(laser)

        # determine if we need to add another pulse to our list of points.
        if (center != None):
            cp = np.add(center, (0, CIRCLE_OFFSET))
            # only add the first point detected in a burst.
            if (prevState == 0):
                pts.append(cp)
                prevState = 1
            cv2.circle(frame, cp, CIRCLE_DIA, CIRCLE_COLOR, CIRCLE_RAD, cv2.LINE_AA)
        else:
            if (prevState == 1):
                prevState = 0

        # loop through pts.
        for pt in pts:
            cv2.circle(frame, pt, CIRCLE_DIA, CIRCLE_COLOR, CIRCLE_RAD, cv2.LINE_AA)
        
        cv2.imshow('Track Laser', frame)
        #cv2.imshow('Track Laser', hsv)

        # check for key presses from user.
        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == ord('q'):
            break;
        elif pressedKey == ord('c'):
            pts.clear()

    cap.release()
    cv2.destroyAllWindows()

# run the main function.
main()