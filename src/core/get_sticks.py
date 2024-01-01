# https://github.com/Rabbid76/python_windows_joystickapi
from utils import joystickapi
import numpy as np
import os
import json
from time import time, sleep, perf_counter
import matplotlib.pyplot as plt
# from drawnow import drawnow
# from tqdm import tqdm


def json_writer(dict_to_write, full_path):
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(dict_to_write, f, ensure_ascii=False, indent=4)


def json_reader(path):
    with open(path) as f:
        data = json.load(f)
    return data


class Joystick:
    def __init__(self):
        num = joystickapi.joyGetNumDevs()
        ret, caps, startinfo = False, None, None
        for id_num in range(num):
            ret, caps = joystickapi.joyGetDevCaps(id_num)
            if ret:
                print("gamepad detected: " + caps.szPname)
                ret, self.startinfo = joystickapi.joyGetPosEx(id_num)
                break
        else:
            raise ModuleNotFoundError("no gamepad detected")
        self.id_num = id_num
        self.btns = None
        self.ret = ret
        self.caps = caps
        self.calib = False

    @property
    def status(self):
        return self.ret

    def read_old(self, with_buttons=False):
        ret, info = joystickapi.joyGetPosEx(self.id_num)
        self.axisXYZ = [info.dwXpos - self.startinfo.dwXpos, info.dwYpos - self.startinfo.dwYpos,
                        info.dwZpos - self.startinfo.dwZpos]
        self.axisRUV = [info.dwRpos - self.startinfo.dwRpos, info.dwUpos - self.startinfo.dwUpos,
                        info.dwVpos - self.startinfo.dwVpos]
        if with_buttons:
            self.btns = [(1 << i) & info.dwButtons != 0 for i in range(self.caps.wNumButtons)]
        return {"axes": [*self.axisXYZ, *self.axisRUV], "buttons": self.btns}

    def read(self):
        ret, info = joystickapi.joyGetPosEx(self.id_num)
        # self.axisXYZ = [info.dwXpos, info.dwYpos, info.dwZpos]
        # self.axisRUV = [info.dwRpos, info.dwUpos, info.dwVpos]
        # return np.array([[*self.axisXYZ, *self.axisRUV]])
        return np.array([[info.dwXpos, info.dwYpos, info.dwZpos, info.dwRpos, info.dwUpos, info.dwVpos]])

    def make_fig_bars(self):
        # self.read_old(with_buttons=False)
        k = 1
        if self.btns is not None:
            k = 2
            plt.subplot(k, 1, 2)
            plt.imshow(np.array(self.btns, dtype=int).reshape(1, -1))
        plt.subplot(k, 1, 1)
        plt.bar(['X', 'Y', 'Z', 'R', 'U', 'V'], [*self.axisXYZ, *self.axisRUV])
        # plt.ylim(-32767, 32767)
        plt.ylim(0, 65535)

    def make_fig_axes(self):
        alpha = 0.2
        plt.subplot(1, 3, 1)
        plt.plot([-1, 1], [0, 0], 'b', lw=3, alpha=alpha)
        plt.plot([0, 0], [-1, 1], 'b', lw=3, alpha=alpha)
        plt.scatter(self.calib_reading[self.sticks["Yaw"]["idx"]], self.calib_reading[self.sticks["Throttle"]["idx"]])
        plt.axis('square')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.subplot(1, 3, 2)
        plt.plot([-1, 1], [0, 0], 'b', lw=3, alpha=alpha)
        plt.plot([0, 0], [-1, 1], 'b', lw=3, alpha=alpha)
        plt.scatter(self.calib_reading[self.sticks["Roll"]["idx"]], self.calib_reading[self.sticks["Pitch"]["idx"]])
        plt.axis('square')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.subplot(1, 3, 3)
        plt.bar(list(self.switches.keys()),
                [self.calib_reading[self.switches["AUX1"]["idx"]], self.calib_reading[self.switches["AUX2"]["idx"]]])
        plt.ylim(-1, 1)

    def render_bars(self):
        drawnow(self.make_fig_bars)

    def render_axes(self):
        drawnow(self.make_fig_axes)

    def calibrate(self, calibration_file_path, load_calibration_file=True):
        def record(t_sec, rps=100, text=None):
            if text is not None:
                print(text)
            readings = self.read()
            for _ in tqdm(range(t_sec * rps)):
                readings = np.vstack((readings, self.read()))
                sleep(1 / rps)
            return readings

        def norm_record(t_sec, rps=100, text=None):
            if text is not None:
                print(text)
            readings = self.norm_read()
            for _ in tqdm(range(t_sec * rps)):
                readings = np.vstack((readings, self.norm_read()))
                sleep(1 / rps)
            return readings

        def get_center(readings):
            for i in range(2, len(readings)):
                if readings[-i:].std(axis=0).mean() > 1e-16:
                    break
            return readings[-i + 1:].mean(axis=0, keepdims=True)

        if load_calibration_file and os.path.exists(calibration_file_path):
            calib_file = json_reader(calibration_file_path)
            # self.active_axes = np.array(calib_file["active_axes"])
            self.min_vals = np.array(calib_file["min_vals"])
            self.max_vals = np.array(calib_file["max_vals"])
            self.sticks = calib_file["sticks"]
            self.switches = calib_file["switches"]
            self.sign_reverse = calib_file["sign_reverse"]

        elif load_calibration_file and not os.path.isfile(calibration_file_path):
            raise FileNotFoundError(
                "Calibration file does not exist. Calibration path given: {}".format(calibration_file_path))

        # Calibrate
        else:
            # Check which sticks move
            print("Move the sticks around, all the way to the edges.")
            readings_axes = record(t_sec=4)
            readings_axes = readings_axes[1:]
            readings_axes_std = readings_axes.std(axis=0)
            if np.any(readings_axes_std > 1e-16):
                self.active_axes = np.sort(np.argsort(readings_axes_std)[::-1][:4])
            else:
                raise ValueError("No sticks detected. Please move the sticks around and try again.")
            print("Axes indices: ", self.active_axes)

            record(t_sec=2, text="center all sticks")

            print("Move the switches all the way (can read only 2 configured switches).")
            readings_swiches = record(t_sec=3)
            readings_swiches = readings_swiches[1:]
            readings_swiches_std = readings_swiches.std(axis=0)
            if np.any(readings_swiches_std > 1e-16):
                self.active_swiches = np.sort(np.argsort(readings_swiches_std)[::-1][:2])
            else:
                raise ValueError("No sticks detected. Please move the sticks around and try again.")
            print("Swiches indices: ", self.active_swiches)

            readings = np.vstack((readings_axes, readings_swiches))
            # Get min and max
            self.min_vals = readings.min(axis=0)
            self.max_vals = readings.max(axis=0)
            self.sign_reverse = np.ones(6)
            # switches_center = (self.min_vals[self.active_swiches] + self.max_vals[self.active_swiches]) / 2 - self.min_vals[self.active_swiches]
            readings = norm_record(t_sec=2, text="center all sticks")
            center = get_center(readings)

            self.sticks = {"Throttle": {}, "Yaw": {}, "Pitch": {}, "Roll": {}}
            commands = ["up", "to the right"]
            for i, k in enumerate(self.sticks.keys()):
                readings = norm_record(t_sec=5, text="\nMove the " + k + " stick " + commands[i % 2])
                idx = self.active_axes[np.argmax(readings[:, self.active_axes].std(axis=0))]
                print("\n" + k + " axis idx: ", idx)
                self.sticks[k]["idx"] = int(idx)
                # self.sticks[k]["sign_reversed"] = np.sign(readings[np.argmax(np.abs(readings[:, idx])), idx])
                self.sign_reverse[idx] = np.sign(readings[np.argmax(np.abs(readings[:, idx])), idx])
                readings = norm_record(t_sec=3, text="center all sticks")
                center = np.vstack((center, get_center(readings)))
            print(self.sticks)
            center = center.mean(axis=0)
            for i, k in enumerate(self.sticks.keys()):
                self.sticks[k]["center"] = center[self.sticks[k]["idx"]]

            self.switches = {"AUX1": {}, "AUX2": {}}
            commands = ["off", "on"]
            for i, k in enumerate(self.switches.keys()):
                # Identify switches
                readings = norm_record(t_sec=4, text=f"Turn {k} on and off repeatedly.")
                idx = self.active_swiches[np.argmax(readings[:, self.active_swiches].std(axis=0))]
                print(k + " axis idx: ", idx)
                self.switches[k]["idx"] = int(idx)
                cond = 0
                attempts = 0
                while cond == 0:
                    # Check switches direction
                    command_reading = {}
                    for command in commands:
                        readings = norm_record(t_sec=3, text=f"\nTurn {k} {command}.")
                        command_reading[command] = readings[-1, idx]
                    norm_record(t_sec=3, text=f"\nTurn {k} off.")
                    cond = command_reading["on"] - command_reading["off"]
                    # self.switches[k]["sign_reversed"] = np.sign(cond)
                    self.sign_reverse[idx] = np.sign(cond)
                    attempts += 1
                    if attempts > 1:
                        print("\nCould not identify switch direction. Please try again.")
                    if attempts > 3:
                        raise ValueError("Could not identify switch direction after 3 attempts.")

            self.save_calibration(dict_to_write={
                # "active_axes": self.active_axes.tolist(),
                "sticks": self.sticks,
                "switches": self.switches,
                "min_vals": self.min_vals.tolist(),
                "max_vals": self.max_vals.tolist(),
                "sign_reverse": self.sign_reverse.tolist()
            }, full_path=calibration_file_path)
        self.calib = True

    def update(self, sticks, switches, min_vals, max_vals, sign_reverse):
        self.sticks = sticks
        self.switches = switches
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.sign_reverse = sign_reverse
        self.calib = True

    def save_calibration(self, dict_to_write, full_path):
        os.path.exists(os.path.dirname(full_path)) or os.makedirs(os.path.dirname(full_path))
        json_writer(dict_to_write, full_path)

    def load_calibration(self, calibration_file_path):
        calib_file = json_reader(calibration_file_path)
        self.min_vals = np.array(calib_file["min_vals"])
        self.max_vals = np.array(calib_file["max_vals"])
        self.sticks = calib_file["sticks"]
        self.switches = calib_file["switches"]
        self.sign_reverse = calib_file["sign_reverse"]

    @staticmethod
    def mapFromTo(x, a, b, c, d):
        y = (x - a) / (b - a) * (d - c) + c
        return y

    def norm_read(self):
        return np.array(
            [[self.mapFromTo(r, self.min_vals[i], self.max_vals[i], -1, 1) for i, r in enumerate(self.read()[0])]])

    def calib_read(self):
        # t0 = time()
        self.calib_reading = self.norm_read()[0] * self.sign_reverse
        for i, k in enumerate(self.sticks.keys()):
            # self.calib_reading[self.sticks[k]["idx"]] *= self.sticks[k]["sign_reversed"]
            if self.calib_reading[self.sticks[k]["idx"]] <= self.sticks[k]["center"]:
                self.calib_reading[self.sticks[k]["idx"]] = self.mapFromTo(self.calib_reading[self.sticks[k]["idx"]],
                                                                           -1, self.sticks[k]["center"], -1, 0)
            else:
                self.calib_reading[self.sticks[k]["idx"]] = self.mapFromTo(self.calib_reading[self.sticks[k]["idx"]],
                                                                           self.sticks[k]["center"], 1, 0, 1)
        return self.calib_reading


if __name__ == '__main__':

    print("start")

    rc = Joystick()
    run = rc.status
    rc.calibrate(os.path.join("config", "frsky.json"), load_calibration_file=True)
    while run:
        # t0 = perf_counter()
        # for i in range(100):
        #     a = rc.calib_read()
        #     if a is None:
        #         break
        # print(1/((perf_counter() - t0) / 100) , "readings per second")
        print(rc.calib_read())
        rc.render_axes()

    """
    run = ret
    while run:
        # time.sleep(0.1)
        if msvcrt.kbhit() and msvcrt.getch() == chr(27).encode(): # detect ESC
            run = False

        ret, info = joystickapi.joyGetPosEx(id)
        if ret:
            btns = [(1 << i) & info.dwButtons != 0 for i in range(caps.wNumButtons)]
            axisXYZ = [info.dwXpos-startinfo.dwXpos, info.dwYpos-startinfo.dwYpos, info.dwZpos-startinfo.dwZpos]
            axisRUV = [info.dwRpos-startinfo.dwRpos, info.dwUpos-startinfo.dwUpos, info.dwVpos-startinfo.dwVpos]
            # if info.dwButtons:
            #     print("buttons: ", btns)
            # if any([abs(v) > 10 for v in axisXYZ]):
            #     print("axis:", axisXYZ)
            # if any([abs(v) > 10 for v in axisRUV]):
            #     print("roation axis:", axisRUV)
            drawnow(make_fig)
    """
    print("end")