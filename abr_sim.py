import os
import numpy as np
from Arguments import get_args
import load_data
import Net_sim
from ABR import VCGTA, Bola, RB, Pensieve, Dynamic, tileSet
from FoV import get_fov
import time

args = get_args()


def main(abr):
    total_tiles = args.total_tiles
    tiles = []
    for i in range(total_tiles):
        tiles.append(tileSet.tile(i, load_data.load_multi_tiled_segment_size(args.movie, i), total_tiles))

    all_timestamp, all_bandwidth, all_recordname = load_data.load_multi_throughput()
    net_model = Net_sim.NetworkModel(all_timestamp, all_bandwidth, all_recordname)

    seg_count = tiles[0].seg_count
    bitrate_level = tiles[0].bitrate

    detail_flag = args.details

    tile_quality_record = []
    tile_bitrate_record = np.zeros((seg_count, total_tiles + 1)).tolist()
    tile_throughput_record = []
    tile_rebuf_duration = []
    tile_buffer = []
    tile_buffer_record = np.zeros((seg_count, total_tiles)).tolist()
    tile_rebuf_times = []
    tile_deviation = []
    tile_face_pri = []
    tile_deviation_record = np.zeros((seg_count, total_tiles)).tolist()
    tile_face_pri_record = np.zeros((seg_count, total_tiles)).tolist()
    tile_dl_size_record = np.zeros((seg_count, total_tiles)).tolist()
    fov_quality_record = np.zeros((seg_count, total_tiles))
    fov_quality_changes = np.zeros((seg_count, total_tiles))
    pre_fov_quality_record = np.zeros((seg_count, total_tiles))
    pre_fov_quality_changes = np.zeros((seg_count, total_tiles))

    pre_pos, act_pos = get_fov.get_total_fov()

    tic = time.time()
    for i in range(tiles[0].seg_count):
        if detail_flag:
            print('segment ' + str(i))
        face_pri = np.repeat(-1, total_tiles).tolist()
        face_pri_real = np.repeat(-1, total_tiles).tolist()

        for j in range(total_tiles):
            tiles[j].compute_deviation(pre_pos[i], act_pos[i])

        for j in range(total_tiles):
            priority_tmp, priority_tmp_real = Net_sim.compute_face_priority(tiles, j)
            tiles[j].update_face_priority(priority_tmp, priority_tmp_real)
            face_pri[j] = priority_tmp
            face_pri_real[j] = priority_tmp_real

        if detail_flag:
            print('Predicted FoV: ' + str(pre_pos[i]))
            print('Actual FoV: ' + str(act_pos[i]))
            print('Face Priority: ' + str(face_pri))

        if abr == 'VCGTA':
            predicted_throughput = VCGTA.pre_bandwidth(tiles, i)
            quality = VCGTA.get_quality_all(tiles, predicted_throughput, net_model.get_all_buffer())

        else:
            quality = np.random.randint(4, size=total_tiles).tolist()

        if detail_flag:
            print(quality)

        real_throughput = 0
        for j in range(total_tiles):
            quality[j] = int(quality[j])

            dl_time, dl_size, average_tput, sleep_time, rebuf, buffer = net_model.sim_download(i, tiles[j], quality[j])

            tiles[j].update_history(dl_time, average_tput, sleep_time, rebuf, buffer, quality[j])
            real_throughput += average_tput
            tile_dl_size_record[i][j] = dl_size
            fov_quality_record[i][tiles[j].face_priority_real] = tiles[j].bitrate[quality[j]]
            pre_fov_quality_record[i][tiles[j].face_priority] = tiles[j].bitrate[quality[j]]
            if j == 0 and detail_flag:
                print('time: ' + str(round(dl_time, 2)) + 'ms\t' +
                      'tput: ' + str(round(average_tput/1000, 2)) + 'kbps\t' +
                      'bit: ' + str(round(tiles[j].bitrate[quality[j]]/1000, 2)) + 'kbps\t' +
                      'sleep: ' + str(round(sleep_time/1000, 2)) + 's\t' +
                      'rebuf: ' + str(round(rebuf/1000, 2)) + 's\t' +
                      'buffer: ' + str(round(buffer/1000, 2)) + 's')

    toc = time.time()
    sim_time = toc - tic

    for i in range(1, tiles[0].seg_count):
        for j in range(total_tiles):
            fov_quality_changes[i][j] = np.abs(fov_quality_record[i][j] - fov_quality_record[i-1][j])
            pre_fov_quality_changes[i][j] = np.abs(pre_fov_quality_record[i][j] - pre_fov_quality_record[i-1][j])

    for i in range(total_tiles):
        tile_quality_record.append(tiles[i].quality_record)
        tile_throughput_record.append(tiles[i].throughput_record)
        tile_buffer.append(tiles[i].buffer_record)
        tile_deviation.append(tiles[i].deviation_real_record)
        tile_face_pri.append(tiles[i].face_pri_real_record)
        rebuf_duration_tmp = 0
        rebuf_times_tmp = 0
        for j in range(1, len(tiles[i].rebuf_record)):
            if tiles[i].rebuf_record[j] != 0:
                rebuf_duration_tmp += tiles[i].rebuf_record[j]
                rebuf_times_tmp += 1
        tile_rebuf_times.append(rebuf_times_tmp)
        tile_rebuf_duration.append(rebuf_duration_tmp)

    time_av_tput = np.sum(tile_throughput_record) / seg_count

    for j in range(seg_count):
        for i in range(total_tiles):
            tile_bitrate_record[j][i] = str(bitrate_level[tile_quality_record[i][j]])
            tile_buffer_record[j][i] = format(tile_buffer[i][j], '.0f')
            tile_deviation_record[j][i] = format(tile_deviation[i][j], '.6f')
            tile_face_pri_record[j][i] = str(tile_face_pri[i][j])
            tile_bitrate_record[j][24] += tile_throughput_record[i][j]
        tile_bitrate_record[j][24] = format(tile_bitrate_record[j][24], '.0f')
    print('\n')
    print('ABR: ' + abr)
    print('Movie: ' + args.movie)
    print('Bitrate level: ' + ','.join([str(bit) for bit in bitrate_level]))
    print('Segment count: ' + str(seg_count))
    print('Network Set: ' + args.network.strip('_txt'))
    print('Network trace: ' + net_model.cur_net_name)
    print('Simulation time: ' + format(sim_time, '.3f'))
    print('Network average throughput: ' + format(net_model.get_av_tput(), '.3f'))
    print('Average throughput: ' + format(time_av_tput, '.3f'))
    print('Total download size: ' + format(np.sum(tile_dl_size_record) / 1000000, '.3f') + ' MB')

    print('Quality in actual FoV: ' + format(np.sum(fov_quality_record[:, :4]) / seg_count, '.3f'))
    print('Quality around actual FoV: ' + format(np.sum(fov_quality_record[:, 4:8]) / seg_count, '.3f'))
    print('Quality out actual FoV: ' + format(np.sum(fov_quality_record[:, 8:]) / seg_count, '.3f'))

    print('Quality in predicted FoV: ' + format(np.sum(pre_fov_quality_record[:, :4]) / seg_count, '.3f'))
    print('Quality around predicted FoV: ' + format(np.sum(pre_fov_quality_record[:, 4:8]) / seg_count, '.3f'))
    print('Quality out predicted FoV: ' + format(np.sum(pre_fov_quality_record[:, 8:]) / seg_count, '.3f'))

    print('Quality changes in actual FoV: ' + format(np.sum(fov_quality_changes[:, :4]) / seg_count, '.3f'))
    print('Quality changes around actual FoV: ' + format(np.sum(fov_quality_changes[:, 4:8]) / seg_count, '.3f'))
    print('Quality changes out actual FoV: ' + format(np.sum(fov_quality_changes[:, 8:]) / seg_count, '.3f'))

    print('Quality changes in predicted FoV: ' + format(np.sum(pre_fov_quality_changes[:, :4]) / seg_count, '.3f'))
    print('Quality changes around predicted FoV: ' + format(np.sum(pre_fov_quality_changes[:, 4:8]) / seg_count, '.3f'))
    print('Quality changes out predicted FoV: ' + format(np.sum(pre_fov_quality_changes[:, 8:]) / seg_count, '.3f'))
    print('Rebuffer events: ' + str(np.sum(tile_rebuf_times)))
    print('Rebuffer duration: ' + format(np.sum(tile_rebuf_duration), '.3f'))
    print('\n')

    os.makedirs('../results/', exist_ok=True)
    os.makedirs('../results/performance/', exist_ok=True)
    os.makedirs('../results/chunkLog/', exist_ok=True)
    os.makedirs('../results/fov/', exist_ok=True)
    os.makedirs('../results/bufferLog/', exist_ok=True)
    os.makedirs('../results/performance/' + args.network.strip('_txt') + '_' + str(net_model.trace_idx) + '_' + args.movie, exist_ok=True)
    os.makedirs('../results/chunkLog/' + args.network.strip('_txt') + '_' + str(net_model.trace_idx) + '_' + args.movie, exist_ok=True)
    os.makedirs('../results/bufferLog/' + args.network.strip('_txt') + '_' + str(net_model.trace_idx) + '_' + args.movie, exist_ok=True)
    os.makedirs('../results/fov/' + args.network.strip('_txt') + '_' + str(net_model.trace_idx) + '_' + args.movie, exist_ok=True)

    with open('../results/performance/' + args.network.strip('_txt') + '_' + str(net_model.trace_idx) + '_' + args.movie + '/' + abr + '.txt', 'w') as f1:
        f1.write('Movie: ' + args.movie + '\n')
        f1.write('Bitrate level: ' + ','.join([str(bit) for bit in bitrate_level]) + '\n')
        f1.write('Segment count: ' + str(seg_count) + '\n')
        f1.write('Network Set: ' + args.network.strip('_txt') + '\n')
        f1.write('Network trace: ' + net_model.cur_net_name + '\n')
        f1.write('Simulation time: ' + format(sim_time, '.3f') + '\n')
        f1.write('Network average throughput: ' + format(net_model.get_av_tput(), '.3f') + '\n')
        f1.write('Average throughput: ' + format(time_av_tput, '.3f') + '\n')
        f1.write('Total download size: ' + format(np.sum(tile_dl_size_record) / 1000000, '.3f') + '\n')
        f1.write('Quality in actual FoV: ' + format(np.sum(fov_quality_record[:, :4]) / seg_count, '.3f') + '\n')
        f1.write('Quality around actual FoV: ' + format(np.sum(fov_quality_record[:, 4:8]) / seg_count, '.3f') + '\n')
        f1.write('Quality out actual FoV: ' + format(np.sum(fov_quality_record[:, 8:]) / seg_count, '.3f') + '\n')
        f1.write('Quality in predicted FoV: ' + format(np.sum(pre_fov_quality_record[:, :4]) / seg_count, '.3f') + '\n')
        f1.write('Quality around predicted FoV: ' + format(np.sum(pre_fov_quality_record[:, 4:8]) / seg_count, '.3f') + '\n')
        f1.write('Quality out predicted FoV: ' + format(np.sum(pre_fov_quality_record[:, 8:]) / seg_count, '.3f') + '\n')
        f1.write('Quality changes in actual FoV: ' + format(np.sum(fov_quality_changes[:, :4]) / seg_count, '.3f') + '\n')
        f1.write('Quality changes around actual FoV: ' + format(np.sum(fov_quality_changes[:, 4:8]) / seg_count, '.3f') + '\n')
        f1.write('Quality changes out actual FoV: ' + format(np.sum(fov_quality_changes[:, 8:]) / seg_count, '.3f') + '\n')
        f1.write('Quality changes in predicted FoV: ' + format(np.sum(fov_quality_changes[:, :4]) / seg_count, '.3f') + '\n')
        f1.write('Quality changes around predicted FoV: ' + format(np.sum(fov_quality_changes[:, 4:8]) / seg_count, '.3f') + '\n')
        f1.write('Quality changes out predicted FoV: ' + format(np.sum(fov_quality_changes[:, 8:]) / seg_count, '.3f') + '\n')
        f1.write('Rebuffer events: ' + str(np.sum(tile_rebuf_times)) + '\n')
        f1.write('Rebuffer duration: ' + format(np.sum(tile_rebuf_duration), '.3f') + '\n')

    with open('../results/chunkLog/' + args.network.strip('_txt') + '_' + str(net_model.trace_idx) + '_' + args.movie + '/' + abr + '.txt', 'w') as f2:
        for i in range(seg_count):
            # segment index * tile index
            f2.write(','.join(tile_bitrate_record[i]) + '\n')

    with open('../results/fov/' + args.network.strip('_txt') + '_' + str(net_model.trace_idx) + '_' + args.movie + '/' + abr + '_devi.txt', 'w') as f3:
        for i in range(seg_count):
            f3.write(','.join(tile_deviation_record[i]) + '\n')

    with open('../results/fov/' + args.network.strip('_txt') + '_' + str(net_model.trace_idx) + '_' + args.movie + '/' + abr + '_fp.txt', 'w') as f4:
        for i in range(seg_count):
            f4.write(','.join(tile_face_pri_record[i]) + '\n')

    with open('../results/bufferLog/' + args.network.strip('_txt') + '_' + str(net_model.trace_idx) + '_' + args.movie + '/' + abr + '.txt', 'w') as f5:
        for i in range(seg_count):
            # segment index * tile index
            f5.write(','.join(tile_buffer_record[i]) + '\n')


if __name__ == '__main__':
    ABR = ['VCGTA']
    for abr in ABR:
        main(abr)
