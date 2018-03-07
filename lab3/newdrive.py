from mypicar.front_wheels import Front_Wheels
from mypicar.back_wheels import Back_Wheels
from detector_wrapper import DetectorWrapper
import numpy as np

detector = DetectorWrapper()

front_wheels = Front_Wheels()
back_wheels = Back_Wheels()

back_wheels.speed = 19

try:
    while True:
        success, ret = detector.detect()
        
        if success:
            detector.plot(ret)
            frame, mid, left_f, right_f, ploty, left_x, right_x = ret

            front_wheels.turn_straight()
            back_wheels.forward()

            diff = mid - 300

            angle = diff / 4
            
            print mid, angle
    
            front_wheels.turn_rel(angle)

except KeyboardInterrupt:
    print("KeboardInterrupt Captured")
finally:
    detector.stop()
    back_wheels.stop()
    front_wheels.turn_straight()
