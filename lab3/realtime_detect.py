from detector_wrapper import DetectorWrapper
detector = DetectorWrapper()

success, ret = detector.detect()

if success:
    detector.plot(ret)
    frame, mid_x, left_fit, right_fit, ploty, left_fitx, right_fitx = ret
