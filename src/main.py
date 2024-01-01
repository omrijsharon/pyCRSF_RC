import os
# import a function that can count microseconds
from time import perf_counter_ns

from core.crsf import CRSF, CRSF_TIME_BETWEEN_FRAMES_US
from core.get_sticks import Joystick
from utils.helper_functions import read_gamepad_and_map2channels, get_port

if __name__ == '__main__':
    last_frame_time = perf_counter_ns()
    crsf = CRSF(get_port(vid=6790))  # Change this to your serial port

    rc = Joystick()
    run = rc.status
    rc.calibrate(os.path.join("config", "frsky.json"), load_calibration_file=True)
    while run:
        current_time = perf_counter_ns()
        if current_time - last_frame_time > CRSF_TIME_BETWEEN_FRAMES_US * 1000:
            channels = read_gamepad_and_map2channels(rc, channel_map='AETR')
            data_packet = crsf.crsf_prepare_data_packet(channels)
            crsf.crsf_write_packet(data_packet)
            last_frame_time = 1 * current_time