import json
import os
from Arguments import get_args

args = get_args()


def load_multi_throughput():
    net_path = args.network_path + '/' + args.network
    throughput_files = os.listdir(net_path)
    all_total_time = []
    all_total_throughput = []
    all_total_record = []

    for file in throughput_files:
        file_path = net_path + '/' + file
        total_time = []
        total_throughput = []
        with open(file_path, 'r') as load_file:
            for line in load_file:
                t = line.strip('\n').split(',')
                t[0] = float(t[0])
                for i in range(1, len(t)):
                    t[i] = float(t[i])
                total_time.append(t[0])
                total_throughput.append(t[-24:])
        all_total_time.append(total_time)
        all_total_throughput.append(total_throughput)
        all_total_record.append(file.strip('.txt'))

    return all_total_time, all_total_throughput, all_total_record


def load_bitrates(movie):
    with open(args.movie_path + movie + '.json', 'r') as file:
        load_movie = json.load(file)
    bitrate = load_movie['bitrate_bps']
    for i in range(len(bitrate)):
        bitrate[i] = bitrate[i] / 1000
    return


def load_per_tile_bitrates(movie):
    with open(args.movie_path + movie + '.json', 'r') as file:
        load_movie = json.load(file)
    bitrate = load_movie['bitrate_per_tile_bps']
    for i in range(len(bitrate)):
        for j in range(len(bitrate[i])):
            for k in range(len(bitrate[i][j])):
                bitrate[i][j][k] = bitrate[i][j][k] / 1000
    return bitrate


def load_segment_size(movie):
    with open(args.movie_path + movie + '.json', 'r') as file:
        load_movie = json.load(file)
    return load_movie['segment_size_bytes']


def load_multi_tiled_segment_size(movie, tile_id):
    with open(args.movie_path + movie + '.json', 'r') as file:
        load_movie = json.load(file)
    tmp = load_movie['segment_size_bytes']
    face = tile_id // 4
    tile_in_face = tile_id % 4
    total = []
    for i in range(len(tmp)):
        total.append(tmp[i][face][tile_in_face])
    return total
