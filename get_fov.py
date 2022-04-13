import numpy as np
import Arguments

args = Arguments.get_args()

def get_fov(seg_idx):
    fov_trace_path = args.fov_trace
    actual_fov = np.loadtxt(fov_trace_path + 'actualFoV.txt')
    predict_fov = np.loadtxt(fov_trace_path + 'predictFoV.txt')

    return predict_fov[seg_idx], actual_fov[seg_idx]


def get_total_fov():
    fov_trace_path = args.fov_trace
    actual_fov = np.loadtxt(fov_trace_path + 'actualFoV.txt')
    predict_fov = np.loadtxt(fov_trace_path + 'predictFoV.txt')

    return predict_fov, actual_fov
