import time
from mypicar.front_wheels import Front_Wheels
from mypicar.back_wheels import Back_Wheels
from detector_wrapper import DetectorWrapper
import numpy as np

detector = DetectorWrapper()

front_wheels = Front_Wheels()
back_wheels = Back_Wheels()

try:
    while True:
        success, ret = detector.detect()
        if success:
            detector.plot(ret)
            frame, mid, left_f, right_f, ploty, left_x, right_x = ret

            front_wheels.turn_straight()
            back_wheels.speed = 17
            back_wheels.forward()

            s1, r2 = detector.detect()

            if s1:
                if r2[1] != mid:
                    diff = r2[1] - mid
                    angle = 0

                    if diff > 20:
                        angle = np.degrees(np.arctan(diff))
                    elif diff < -20:
                        angle = -np.degrees(np.arctan(diff))

                    front_wheels.turn_rel(angle)
                    back_wheels.forward()
                    time.sleep(2)
                    front_wheels.turn_straight()
                    back_wheels.forward()

except KeyboardInterrupt:
    print("KeboardInterrupt Captured")
finally:
    detector.stop()
    back_wheels.stop()
    front_wheels.turn_straight()
