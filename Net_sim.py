import numpy as np
import load_data
from Arguments import get_args

args = get_args()
RANDOM_SEED = 42
BIT_IN_BYTE = 8
K_IN_M = 1000
MS_IN_S = 1000
LINK_RTT = 80   # ms
PACKET_PAYLOAD_PORTION = 0.95
DRAIN_BUFFER_SLEEP_TIME = 500.0  # ms


def compute_face_priority(tilelist, tile_id):
    face_rank = len(tilelist) - 1
    face_rank_real = len(tilelist) - 1
    for i in range(len(tilelist)):
        if tilelist[i].deviation > tilelist[tile_id].deviation:
            face_rank -= 1
        elif tilelist[i].deviation == tilelist[tile_id].deviation and i > tile_id:
            face_rank -= 1
        if tilelist[i].deviation_real > tilelist[tile_id].deviation_real:
            face_rank_real -= 1
        elif tilelist[i].deviation_real == tilelist[tile_id].deviation_real and i > tile_id:
            face_rank_real -= 1
    return face_rank, face_rank_real

class NetworkModel:
    def __init__(self, net_time, net_bw, net_record, random_seed=RANDOM_SEED):
        assert len(net_time) == len(net_bw)
        self.all_cooked_net_time = net_time
        self.all_cooked_net_bw = net_bw
        self.all_cooked_net_name = net_record

        if args.network_trace == -1:
            self.trace_idx = np.random.randint(len(self.all_cooked_net_time))
        else:
            self.trace_idx = args.network_trace
        self.cur_net_time = self.all_cooked_net_time[self.trace_idx]
        self.cur_net_bw = self.all_cooked_net_bw[self.trace_idx]
        self.cur_net_name = self.all_cooked_net_name[self.trace_idx]

        self.all_seg_idx = np.zeros(args.total_tiles).tolist()
        self.all_buffer = np.zeros(args.total_tiles).tolist()

        self.timestamp_pre_idx = 1
        self.timestamp_last_idx = self.timestamp_pre_idx - 1
        self.timestamp_pre = self.cur_net_time[self.timestamp_pre_idx]
        self.timestamp_last = self.cur_net_time[self.timestamp_pre_idx - 1]
        self.all_ts_pre_idx = np.repeat(np.expand_dims(self.timestamp_pre_idx, axis=0), args.total_tiles, axis=0)
        self.all_ts_last = np.repeat(np.expand_dims(self.timestamp_last, axis=0), args.total_tiles, axis=0)

        self.video_size = load_data.load_segment_size(args.movie)
        self.seg_duration = args.segment_duration * MS_IN_S
        self.max_buffer = args.buffer_size * MS_IN_S

    def get_buffer(self, tile_idx):
        return self.all_buffer[tile_idx]

    def get_all_buffer(self):
        return self.all_buffer

    def get_av_tput(self):
        return np.sum(self.cur_net_bw) / len(self.cur_net_bw)

    def sim_download(self, seg_idx, tile, quality):
        assert quality >= 0
        assert quality < len(tile.bitrate)
        assert seg_idx < tile.seg_count

        tile_id = tile.tile_id
        face_idx = tile_id // 4
        tile_in_face = tile_id % 4
        segment_size = self.video_size[seg_idx][face_idx][tile_in_face][quality]

        dl_time = 0
        downloaded_size = 0
        sleep_time = 0

        if self.all_buffer[tile_id] > self.max_buffer - self.seg_duration:
            sleep_time += np.ceil((self.all_buffer[tile_id] - self.max_buffer + self.seg_duration)
                                  / DRAIN_BUFFER_SLEEP_TIME) * DRAIN_BUFFER_SLEEP_TIME
            self.all_buffer[tile_id] -= sleep_time
            sleep_time_ = sleep_time
            while True:
                sleep_duration = self.cur_net_time[self.all_ts_pre_idx[tile_id]] \
                                 - self.all_ts_last[tile_id]
                if sleep_duration > sleep_time_:
                    self.all_ts_last[tile_id] += sleep_time_
                    break
                sleep_time_ -= sleep_duration
                self.all_ts_last[tile_id] = self.cur_net_time[self.all_ts_pre_idx[tile_id]]
                self.all_ts_pre_idx[tile_id] += 1

                if self.all_ts_pre_idx[tile_id] >= len(self.cur_net_time):
                    self.all_ts_pre_idx[tile_id] = 1
                    self.all_ts_last[tile_id] = 0

        while True:
            tput = self.cur_net_bw[self.all_ts_pre_idx[tile_id]][tile_id] / BIT_IN_BYTE
            tput_duration = (self.cur_net_time[self.all_ts_pre_idx[tile_id]]
                             - self.all_ts_last[tile_id]) / MS_IN_S

            next_packet = tput * tput_duration * PACKET_PAYLOAD_PORTION
            if next_packet > (segment_size - downloaded_size):
                fractional_time = (segment_size - downloaded_size) / tput / PACKET_PAYLOAD_PORTION
                dl_time += fractional_time
                self.all_ts_last[tile_id] += fractional_time
                break
            dl_time += tput_duration
            downloaded_size += next_packet
            self.all_ts_last[tile_id] = self.cur_net_time[self.all_ts_pre_idx[tile_id]]
            self.all_ts_pre_idx[tile_id] += 1

            if self.all_ts_pre_idx[tile_id] >= len(self.cur_net_bw):
                self.all_ts_pre_idx[tile_id] = 1
                self.all_ts_last[tile_id] = 0

        dl_time *= MS_IN_S
        dl_time += LINK_RTT
        dl_time *= np.random.uniform(0.9, 1.1)

        average_tput = segment_size / dl_time * BIT_IN_BYTE * MS_IN_S

        rebuf = max(dl_time - self.all_buffer[tile_id], 0)

        self.all_buffer[tile_id] += self.seg_duration

        self.all_buffer[tile_id] = max(self.all_buffer[tile_id] - dl_time, 0)

        return dl_time, segment_size, average_tput, sleep_time, rebuf, self.all_buffer[tile_id]
