import argparse
import threading
import os
import json
import time
import logging
import motioncapture

from datetime import datetime
from log import LoggerFactory


# The host name or ip address of the mocap system
host_name = '192.168.1.39'

# The type of the mocap system
# Valid options are: 'vicon', 'optitrack', 'optitrack_closed_source', 'qualisys', 'nokov', 'vrpn', 'motionanalysis'
mocap_system_type = 'vicon'


class MocapWrapper(threading.Thread):
    def __init__(self, body_name, log_level=logging.INFO):
        threading.Thread.__init__(self)

        self.body_name = body_name
        self.on_pose = None
        self._stay_open = True
        self.all_frames = []
        self.logger = LoggerFactory("Mocap", level=log_level).get_logger()

        self.start()

    def close(self):
        self._stay_open = False

        now = datetime.now()
        formatted = now.strftime("%H_%M_%S_%m_%d_%Y")
        file_path = os.path.join("logs", f"{mocap_system_type}_{formatted}.json")
        with open(file_path, "w") as f:
            json.dump({"frames": self.all_frames}, f)
        self.logger.info(f"Mocap log saved in {file_path}")
        self.join()

    def run(self):
        mc = motioncapture.connect(mocap_system_type, {'hostname': host_name})
        i = 0
        while self._stay_open:
            mc.waitForNextFrame()
            for name, obj in mc.rigidBodies.items():
                if name == self.body_name:
                    now = time.time()
                    pos = obj.position
                    if self.on_pose:
                        self.on_pose([pos[0], pos[1], pos[2], obj.rotation, now])
                    self.all_frames.append({
                        "frame_id": i,
                        "tvec": [float(pos[0]), float(pos[1]), float(pos[2])],
                        "time": now * 1000
                    })
                    i += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", default=140, type=int, help="duration")
    args = ap.parse_args()

    mw = MocapWrapper("fls_px4")
    time.sleep(args.t)
    mw.close()
