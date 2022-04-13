from argparse import ArgumentParser

parser = ArgumentParser(description = 'VCGTA')

parser.add_argument('--total_tiles', default=24, type=int)
parser.add_argument('--buffer_size', default=25, type=int)
parser.add_argument('--segment_duration', default=1, type=int)
parser.add_argument('--movie', default='tiledGate', type=str)
parser.add_argument('--network', default='4Glogs_txt', type=str)
parser.add_argument('--network_trace', default=0, type=int)
parser.add_argument('-d', '--details', action='store_true')
parser.add_argument('--quality_factor', default=0.001, type=int)
parser.add_argument('--intra_smooth_factor', default=0.001, type=int)
parser.add_argument('--inter_smooth_factor', default=0.001, type=int)
parser.add_argument('--rebuf_factor', default=0.001, type=int)
parser.add_argument('--fov_path', default='../Data/FoV/', type=str)
parser.add_argument('--fov_trace', default='../Data/FoV_trajectory/', type=str)
parser.add_argument('--movie_path', default='../Data/movie/', type=str)
parser.add_argument('--network_path', default='../Data/throughput/', type=str)

parser.add_argument('--input_size', default=5, type=int, help='input size size for ConvLSTM')
parser.add_argument('--hidden_size', default=8, type=int, help='hidden size for ConvLSTM')
parser.add_argument('--num_layers', default=1, type=int, help='layers for ConvLSTM')
parser.add_argument('--learning_rate', default=1e-3, type=float,  help='learning rate of model')
parser.add_argument('--epochs', default=70, type=int, help='train epochs')

parser.add_argument('--start_frame', default=0, type=int, help = 'started frameId')
parser.add_argument('--blur_size_width', default=5, type=int, help = 'blur_size_width')
parser.add_argument('--blur_size_high', default=5, type=int, help = 'blur_size_high')
parser.add_argument('--windows', '-w', default=4, type=int, help='prediction window size')
parser.add_argument('--threshold', default=0.2, type=float, help='control predict_tile choose')
parser.add_argument('--showImage', default=False, type=bool, help='show test image or not')

# path setting
parser.add_argument('--save_path', default = './model/', type=str, metavar='PATH', help='path to save online model')
parser.add_argument('--log_path', default = './ABRFoV', type=str, metavar='PATH', help='log base path')

def get_args():
    args = parser.parse_args()
    return args

