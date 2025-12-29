import cv2
import numpy as np
import pygame

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

# NEW: live-adjustable offsets (pixels)
X_OFFSET = 0      # right (+), left (-)
Y_OFFSET = 0      # down (+), up (-)

# NEW: how much arrow keys change the offset each press
OFFSET_STEP = 1

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


def draw_overlay(frame, x_off, y_off, step):
    """
    Draw offset info near the top of the screen.
    """
    text = f"X_OFFSET: {x_off:+d}   Y_OFFSET: {y_off:+d}   STEP: {step}"
    # shadow for readability
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    help_text = "Arrows=adjust  [/] step  r=reset  c=clear  q=quit"
    cv2.putText(frame, help_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, help_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    global prevState
    global pts
    global cap
    global X_OFFSET, Y_OFFSET, OFFSET_STEP

    cap = cv2.VideoCapture(0)

    pygame.mixer.init()
    hit_sound = pygame.mixer.Sound("media/hit.wav")

    while (1):
        # Take each frame
        ret, frame = cap.read()
        if not ret:
            break

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
        if (center is not None):
            # NEW: apply live offsets + your existing CIRCLE_OFFSET
            cp = (
                center[0] + X_OFFSET,
                center[1] + CIRCLE_OFFSET + Y_OFFSET
            )

            # only add the first point detected in a burst.
            if (prevState == 0):
                pts.append(cp)
                prevState = 1
                pygame.mixer.Sound.play(hit_sound)
        else:
            if (prevState == 1):
                prevState = 0

        # loop through pts.
        for pt in pts:
            cv2.circle(frame, pt, CIRCLE_DIA, CIRCLE_COLOR, CIRCLE_RAD, cv2.LINE_AA)

        # NEW: overlay offset values on-screen
        draw_overlay(frame, X_OFFSET, Y_OFFSET, OFFSET_STEP)

        cv2.imshow('Track Laser', frame)

        # NEW: use waitKeyEx to reliably read arrow keys on Windows
        key = cv2.waitKeyEx(1)

        # Quit
        if key == ord('q'):
            break

        # Clear points
        elif key == ord('c'):
            pts.clear()

        # Reset offsets
        elif key == ord('r'):
            X_OFFSET = 0
            Y_OFFSET = 0

        # Step size adjust
        elif key == ord('['):
            OFFSET_STEP = max(1, OFFSET_STEP // 2)
        elif key == ord(']'):
            OFFSET_STEP = min(100, OFFSET_STEP * 2)

        # Arrow keys (common OpenCV codes)
        # Left=2424832, Up=2490368, Right=2555904, Down=2621440 (waitKeyEx on Windows)
        elif key == 2424832:   # Left
            X_OFFSET -= OFFSET_STEP
        elif key == 2555904:   # Right
            X_OFFSET += OFFSET_STEP
        elif key == 2490368:   # Up
            Y_OFFSET -= OFFSET_STEP
        elif key == 2621440:   # Down
            Y_OFFSET += OFFSET_STEP

    cap.release()
    cv2.destroyAllWindows()


# run the main function.
main()
