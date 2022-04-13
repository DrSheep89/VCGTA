import math
import numpy as np


class tile:
    def __init__(self, tile_id, seg_size, total_tiles):
        if total_tiles == 6:
            FACE_COORD = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1], [0, 0, 1]]
        elif total_tiles == 24:
            FACE_COORD = [[1, 0.5, -0.5], [1, 0.5, 0.5], [1, -0.5, -0.5], [1, -0.5, 0.5],
                          [-1, 0.5, 0.5], [-1, 0.5, -0.5], [-1, -0.5, 0.5], [-1, -0.5, -0.5],
                          [-0.5, 1, 0.5], [0.5, 1, 0.5], [-0.5, 1, -0.5], [0.5, 1, -0.5],
                          [-0.5, -1, -0.5], [0.5, -1, -0.5], [-0.5, -1, 0.5], [0.5, -1, 0.5],
                          [-0.5, 0.5, -1], [0.5, 0.5, -1], [-0.5, -0.5, -1], [0.5, -0.5, -1],
                          [0.5, 0.5, 1], [-0.5, 0.5, 1], [0.5, -0.5, 1], [-0.5, -0.5, 1]]
        else:
            FACE_COORD = []
        self.tile_id = tile_id
        self.seg_size = seg_size
        self.seg_count = len(seg_size)
        self.bitrate = np.round(np.mean(self.seg_size, axis=0), decimals=2)
        self.face_coord = FACE_COORD[tile_id]
        self.quality_record = []
        self.DL_time_record = []
        self.sleep_time_record = []
        self.rebuf_record = []
        self.buffer_record = []
        self.throughput_record = []
        self.deviation_record = []
        self.deviation_real_record = []
        self.face_pri_record = []
        self.face_pri_real_record = []
        self.deviation = 0
        self.dev_weight = 0
        self.deviation_real = 0
        self.dev_weight_real = 0
        self.face_priority = total_tiles
        self.face_priority_real = total_tiles

    def compute_deviation(self, pre_pos, act_pos):
        self.deviation = math.acos(
            (pre_pos[0] * self.face_coord[0] + pre_pos[1] * self.face_coord[1] + pre_pos[2] * self.face_coord[2]) /
            math.sqrt(pre_pos[0] * pre_pos[0] + pre_pos[1] * pre_pos[1] + pre_pos[2] * pre_pos[2]) /
            math.sqrt(self.face_coord[0] * self.face_coord[0] + self.face_coord[1] * self.face_coord[1]
                      + self.face_coord[2] * self.face_coord[2]))
        self.deviation_record.append(self.deviation)
        self.deviation_real = math.acos(
            (act_pos[0] * self.face_coord[0] + act_pos[1] * self.face_coord[1] + act_pos[2] * self.face_coord[2]) /
            math.sqrt(act_pos[0] * act_pos[0] + act_pos[1] * act_pos[1] + act_pos[2] * act_pos[2]) /
            math.sqrt(self.face_coord[0] * self.face_coord[0] + self.face_coord[1] * self.face_coord[1]
                      + self.face_coord[2] * self.face_coord[2]))
        self.deviation_real_record.append(self.deviation_real)

    def update_face_priority(self, face_pri, face_pri_real):
        self.face_priority = face_pri
        if face_pri < 4:
            self.dev_weight = 1
        elif face_pri < 8:
            self.dev_weight = 0.5
        else:
            self.dev_weight = 0
        self.face_pri_record.append(face_pri)

        self.face_priority_real = face_pri_real
        if face_pri_real < 4:
            self.dev_weight_real = 1
        elif face_pri_real < 8:
            self.dev_weight_real = 0.5
        else:
            self.dev_weight_real = 0
        self.face_pri_real_record.append(face_pri_real)

    def update_history(self, dl_time, throughput, sleep_time, rebuf, buffer, quality):
        self.quality_record.append(quality)
        self.DL_time_record.append(dl_time)
        self.rebuf_record.append(rebuf)
        self.buffer_record.append(buffer)
        self.throughput_record.append(throughput)
        self.sleep_time_record.append(sleep_time)

    def get_total_rebuf(self):
        rebuf_time = 0
        rebuf_duration = 0
        for i in range(len(self.rebuf_record)):
            if self.rebuf_record[i] != 0:
                rebuf_time += 1
                rebuf_duration += self.rebuf_record[i]
        return rebuf_time, rebuf_duration

    def get_average_throughput(self, window=5):
        assert window > 0
        if not self.throughput_record:
            return 1e-2
        elif len(self.throughput_record) <= window:
            return np.sum(np.array(self.throughput_record)) / len(self.throughput_record)
        else:
            return np.sum(np.array(self.throughput_record[-window:])) / window
