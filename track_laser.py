import cv2
import numpy as np
import pygame
import time

# ----------------------------
# Drawing / UI constants
# ----------------------------
CIRCLE_DIA = 5
CIRCLE_RAD = 2
CIRCLE_COLOR = (0, 0, 255)      # BGR
BAR_HEIGHT = 28
CIRCLE_OFFSET = BAR_HEIGHT // 2 - 4

# NEW: live-adjustable offsets (pixels)
X_OFFSET = 0      # right (+), left (-)
Y_OFFSET = 0      # down (+), up (-)

# NEW: how much arrow keys change the offset each press
OFFSET_STEP = 1

# ----------------------------
# Laser detection parameters (tunable live)
# ----------------------------
MIN_AREA = 3
MAX_AREA = 300
MIN_CIRCULARITY = 0.35

PERCENTILE_THRESH = 99.55   # higher => fewer false positives, may miss dim dot
MIN_REDNESS_ABS = 10        # floor so it doesn't trigger on noise

KERNEL_SIZE = 3            # morphology kernel size (odd recommended)
MORPH_OPEN_ITERS = 1
MORPH_CLOSE_ITERS = 1

# ----------------------------
# New: detection cooldown (seconds)
# ----------------------------
HIT_COOLDOWN_SEC = 0.06  # 60 ms

# Optional camera / exposure controls (camera-dependent)
ENABLE_EXPOSURE_TWEAKS = False
AUTO_EXPOSURE_VALUE = 0.25   # often "manual" on some Windows drivers
EXPOSURE_VALUE = -6          # varies a lot by camera


# ----------------------------
# Globals
# ----------------------------
cap = None
prevState = 0
pts = []


def findCenterBest(mask, min_area, max_area, min_circ):
    """
    Find the best laser candidate blob by area + circularity.
    Returns (center_xy, contour) or (None, None)
    """
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return None, None

    best_center = None
    best_contour = None
    best_score = -1.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue

        circularity = (4.0 * np.pi * area) / (perim * perim)
        if circularity < min_circ:
            continue

        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            (x, y), _ = cv2.minEnclosingCircle(c)
            cx, cy = int(x), int(y)

        # score: prefer larger + rounder blobs
        score = area * (0.5 + circularity)
        if score > best_score:
            best_score = score
            best_center = (cx, cy)
            best_contour = c

    return best_center, best_contour


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_overlay(frame,
                 x_off, y_off, offset_step,
                 min_area, max_area, min_circ,
                 pct, min_red_abs, ksize,
                 open_it, close_it,
                 overlay_enabled,
                 cooldown_sec):
    """
    Draw offset + detection parameter info and key instructions.
    """
    y = 25
    lh = 22  # line height

    def put(line, y, scale=0.55):
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

    put(f"Overlay: {'ON' if overlay_enabled else 'OFF'} (press 'h' to toggle)   Cooldown: {cooldown_sec:.2f}s", y, 0.60); y += lh
    put(f"Offsets: X_OFFSET {x_off:+d}   Y_OFFSET {y_off:+d}   STEP {offset_step}", y, 0.65); y += lh
    put(f"Detect: MIN_AREA {min_area}  MAX_AREA {max_area}  MIN_CIRC {min_circ:.2f}", y); y += lh
    put(f"Thresh: PCT {pct:.2f}  MIN_RED_ABS {min_red_abs}  KERNEL {ksize}  OPEN {open_it}  CLOSE {close_it}", y); y += lh + 6

    put("Keys: Arrows=move circle  [ ]=offset step  r=reset offsets  c=clear points  q=quit", y); y += lh
    put("Tune: 1/2 MIN_RED_ABS  3/4 PCT  5/6 MIN_AREA  7/8 MAX_AREA  9/0 MIN_CIRC", y); y += lh
    put("Morph: -/= KERNEL   o/p OPEN iters   k/l CLOSE iters   d=toggle debug mask   h=toggle overlay", y); y += lh


def configure_camera(camera):
    """
    Optional exposure tweaks. Works only on some drivers/cameras.
    """
    if not ENABLE_EXPOSURE_TWEAKS:
        return

    try:
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_VALUE)
        camera.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VALUE)
    except Exception:
        pass


def main():
    global cap, prevState, pts
    global X_OFFSET, Y_OFFSET, OFFSET_STEP
    global MIN_AREA, MAX_AREA, MIN_CIRCULARITY
    global PERCENTILE_THRESH, MIN_REDNESS_ABS
    global KERNEL_SIZE, MORPH_OPEN_ITERS, MORPH_CLOSE_ITERS

    cap = cv2.VideoCapture(0)
    configure_camera(cap)

    pygame.mixer.init()
    hit_sound = pygame.mixer.Sound("media/hit.wav")

    debug_show_mask = False
    overlay_enabled = True

    # New: cooldown timing to prevent double-hits on flicker
    last_hit_time = -1e9  # effectively "a long time ago"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Red dominance mask + dynamic threshold ---
        b, g, r = cv2.split(frame)

        # Redness: R - max(G,B) suppresses white glare and non-red highlights
        max_gb = cv2.max(g, b)
        redness = cv2.subtract(r, max_gb)  # uint8 0..255

        # Dynamic threshold: use a high percentile of redness in THIS frame
        thr = float(np.percentile(redness, PERCENTILE_THRESH))
        thr = max(thr, float(MIN_REDNESS_ABS))

        _, laser = cv2.threshold(redness, thr, 255, cv2.THRESH_BINARY)

        # Morphology cleanup
        ksize = max(1, int(KERNEL_SIZE))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        if MORPH_OPEN_ITERS > 0:
            laser = cv2.morphologyEx(laser, cv2.MORPH_OPEN, k, iterations=int(MORPH_OPEN_ITERS))
        if MORPH_CLOSE_ITERS > 0:
            laser = cv2.morphologyEx(laser, cv2.MORPH_CLOSE, k, iterations=int(MORPH_CLOSE_ITERS))

        # Find best candidate dot
        center, contour = findCenterBest(laser, MIN_AREA, MAX_AREA, MIN_CIRCULARITY)

        # Add point on first detection in a burst + cooldown guard
        now = time.monotonic()
        if center is not None:
            # NEW: apply live offsets + your existing CIRCLE_OFFSET
            cp = (
                center[0] + X_OFFSET,
                center[1] + CIRCLE_OFFSET + Y_OFFSET
            )

            # Only add the first point detected in a burst,
            # AND enforce a minimum time between hits to avoid flicker double-hits.
            if prevState == 0 and (now - last_hit_time) >= HIT_COOLDOWN_SEC:
                pts.append(cp)
                prevState = 1
                last_hit_time = now
                pygame.mixer.Sound.play(hit_sound)
            else:
                # still considered "in detection state" even if we didn't log a hit
                prevState = 1
        else:
            prevState = 0

        # Draw points
        for pt in pts:
            cv2.circle(frame, pt, CIRCLE_DIA, CIRCLE_COLOR, CIRCLE_RAD, cv2.LINE_AA)

        # Optional: draw contour/center for debugging
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 1)  # yellow outline
            cv2.circle(frame, center, 3, (0, 255, 255), -1)

        # Overlay (optional)
        if overlay_enabled:
            draw_overlay(frame,
                         X_OFFSET, Y_OFFSET, OFFSET_STEP,
                         MIN_AREA, MAX_AREA, MIN_CIRCULARITY,
                         PERCENTILE_THRESH, MIN_REDNESS_ABS, KERNEL_SIZE,
                         MORPH_OPEN_ITERS, MORPH_CLOSE_ITERS,
                         overlay_enabled,
                         HIT_COOLDOWN_SEC)
        else:
            # Still give a tiny hint so users remember how to bring it back
            cv2.putText(frame, "Overlay OFF (press 'h' to toggle)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "Overlay OFF (press 'h' to toggle)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Track Laser", frame)
        if debug_show_mask:
            cv2.imshow("Laser Mask (debug)", laser)
        else:
            try:
                cv2.destroyWindow("Laser Mask (debug)")
            except Exception:
                pass

        # Key handling (use waitKeyEx for arrows)
        key = cv2.waitKeyEx(1)

        # Quit
        if key == ord('q'):
            break

        # Toggle overlay
        elif key == ord('h'):
            overlay_enabled = not overlay_enabled

        # Clear points
        elif key == ord('c'):
            pts.clear()

        # Reset offsets
        elif key == ord('r'):
            X_OFFSET = 0
            Y_OFFSET = 0

        # Toggle debug
        elif key == ord('d'):
            debug_show_mask = not debug_show_mask

        # Offset step adjust
        elif key == ord('['):
            OFFSET_STEP = max(1, OFFSET_STEP // 2)
        elif key == ord(']'):
            OFFSET_STEP = min(100, OFFSET_STEP * 2)

        # Arrow keys (Windows waitKeyEx codes)
        elif key == 2424832:   # Left
            X_OFFSET -= OFFSET_STEP
        elif key == 2555904:   # Right
            X_OFFSET += OFFSET_STEP
        elif key == 2490368:   # Up
            Y_OFFSET -= OFFSET_STEP
        elif key == 2621440:   # Down
            Y_OFFSET += OFFSET_STEP

        # ----------------------------
        # Detection tuning keys
        # ----------------------------
        # MIN_REDNESS_ABS: 1/2
        elif key == ord('1'):
            MIN_REDNESS_ABS = clamp(MIN_REDNESS_ABS - 2, 0, 255)
        elif key == ord('2'):
            MIN_REDNESS_ABS = clamp(MIN_REDNESS_ABS + 2, 0, 255)

        # PERCENTILE_THRESH: 3/4
        elif key == ord('3'):
            PERCENTILE_THRESH = clamp(PERCENTILE_THRESH - 0.05, 90.0, 99.99)
        elif key == ord('4'):
            PERCENTILE_THRESH = clamp(PERCENTILE_THRESH + 0.05, 90.0, 99.99)

        # MIN_AREA: 5/6
        elif key == ord('5'):
            MIN_AREA = max(0, MIN_AREA - 1)
        elif key == ord('6'):
            MIN_AREA = MIN_AREA + 1

        # MAX_AREA: 7/8
        elif key == ord('7'):
            MAX_AREA = max(MIN_AREA, MAX_AREA - 10)
        elif key == ord('8'):
            MAX_AREA = MAX_AREA + 10

        # MIN_CIRCULARITY: 9/0
        elif key == ord('9'):
            MIN_CIRCULARITY = clamp(MIN_CIRCULARITY - 0.02, 0.0, 1.0)
        elif key == ord('0'):
            MIN_CIRCULARITY = clamp(MIN_CIRCULARITY + 0.02, 0.0, 1.0)

        # KERNEL_SIZE: - / =  (keep odd >= 1)
        elif key == ord('-'):
            KERNEL_SIZE = max(1, KERNEL_SIZE - 2)
        elif key == ord('='):
            KERNEL_SIZE = KERNEL_SIZE + 2

        # Morph iterations
        # OPEN: o/p
        elif key == ord('o'):
            MORPH_OPEN_ITERS = max(0, MORPH_OPEN_ITERS - 1)
        elif key == ord('p'):
            MORPH_OPEN_ITERS = MORPH_OPEN_ITERS + 1

        # CLOSE: k/l
        elif key == ord('k'):
            MORPH_CLOSE_ITERS = max(0, MORPH_CLOSE_ITERS - 1)
        elif key == ord('l'):
            MORPH_CLOSE_ITERS = MORPH_CLOSE_ITERS + 1

        # Ensure MAX_AREA is always >= MIN_AREA
        if MAX_AREA < MIN_AREA:
            MAX_AREA = MIN_AREA

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
