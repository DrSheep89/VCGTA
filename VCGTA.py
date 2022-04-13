import Arguments
import numpy as np

args = Arguments.get_args()


def pre_bandwidth(tilelist, seg_idx):
    total_tiles = args.total_tiles
    if seg_idx == 0:
        return 0
    pre_tput = 0
    if seg_idx <= 5:
        for i in range(total_tiles):
            pre_tput += np.sum(tilelist[i].throughput_record)
        pre_tput /= seg_idx
    else:
        for i in range(total_tiles):
            pre_tput += np.sum(tilelist[i].throughput_record[-5:])
        pre_tput /= 5
    return pre_tput * 1.2


def get_quality_all(tilelist, pre_tput, all_buffer):
    total_tiles = args.total_tiles
    quality_array = np.repeat(0, total_tiles).tolist()
    current_tput = 0

    face_priority = np.repeat(0, total_tiles).tolist()
    for i in range(total_tiles):
        face_priority[i] = tilelist[i].face_priority_real
        current_tput += tilelist[i].bitrate[int(quality_array[i])]

    for i in range(total_tiles):
        face_priority_idx = face_priority.index(i)
        if tilelist[face_priority_idx].face_priority < 6:
            for j in range(1, len(tilelist[face_priority_idx].bitrate)):
                if all_buffer[face_priority_idx] > 1000:
                    if current_tput - tilelist[face_priority_idx].bitrate[int(quality_array[i])] + \
                            tilelist[face_priority_idx].bitrate[j] <= pre_tput:
                        quality_array[face_priority_idx] = j
                        current_tput = current_tput - tilelist[face_priority_idx].bitrate[int(quality_array[i])] + \
                                       tilelist[face_priority_idx].bitrate[j]

        elif tilelist[face_priority_idx].face_priority < 10:
            for j in range(1, len(tilelist[face_priority_idx].bitrate) - 1):
                if all_buffer[face_priority_idx] > 2000:
                    if current_tput - tilelist[face_priority_idx].bitrate[int(quality_array[i])] + \
                            tilelist[face_priority_idx].bitrate[j] <= pre_tput:
                        quality_array[face_priority_idx] = j
                        current_tput = current_tput - tilelist[face_priority_idx].bitrate[int(quality_array[i])] + \
                                       tilelist[face_priority_idx].bitrate[j]

    return quality_array
