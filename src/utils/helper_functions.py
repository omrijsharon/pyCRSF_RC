import serial.tools.list_ports


def print_ports():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(port.name, port.description, port.vid, port.pid, port.hwid)


def get_port(vid=None):
    ports = serial.tools.list_ports.comports()
    if vid is None:
        return ports[0].name
    for port in ports:
        if port.vid == vid:
            return port.name
    return None

# convert self.calib_reading to channel values.
# example: channel[2] = self.calib_reading[self.sticks["Roll"]["idx"]]
# example: self.calib_reading[self.sticks["Yaw"]["idx"]], self.calib_reading[self.sticks["Throttle"]["idx"]]
def read_gamepad_and_map2channels(rc, channel_map='AETR'): # channel map can be ether 'AETR' or 'TAER'
    channels = [1000] * len(rc.calib_reading)
    if channel_map == 'AETR':
        channels[0] = rc.calib_reading[rc.sticks["Roll"]["idx"]]  # Aileron -> Roll
        channels[1] = rc.calib_reading[rc.sticks["Pitch"]["idx"]]  # Elevator -> Pitch
        channels[2] = rc.calib_reading[rc.sticks["Throttle"]["idx"]]
        channels[3] = rc.calib_reading[rc.sticks["Yaw"]["idx"]]  # Rudder -> Yaw
        channels[4] = rc.calib_reading[rc.sticks["AUX1"]["idx"]]  # Aux1 -> AUX1
        channels[5] = rc.calib_reading[rc.sticks["AUX2"]["idx"]]  # Aux2 -> AUX2
        channels[6] = rc.calib_reading[rc.sticks["AUX3"]["idx"]]  # Aux3 -> AUX3
        channels[7] = rc.calib_reading[rc.sticks["AUX4"]["idx"]]  # Aux4 -> AUX4
    elif channel_map == 'TAER':
        channels[0] = rc.calib_reading[rc.sticks["Throttle"]["idx"]]
        channels[1] = rc.calib_reading[rc.sticks["Roll"]["idx"]] # Aileron -> Roll
        channels[2] = rc.calib_reading[rc.sticks["Pitch"]["idx"]] # Elevator -> Pitch
        channels[3] = rc.calib_reading[rc.sticks["Yaw"]["idx"]] # Rudder -> Yaw
        channels[4] = rc.calib_reading[rc.sticks["AUX1"]["idx"]]
        channels[5] = rc.calib_reading[rc.sticks["AUX2"]["idx"]]
        channels[6] = rc.calib_reading[rc.sticks["AUX3"]["idx"]]
        channels[7] = rc.calib_reading[rc.sticks["AUX4"]["idx"]]
    else:
        raise ValueError('channel_map must be either "AETR" or "TAER"')
    return channels