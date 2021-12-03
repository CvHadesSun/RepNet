


import os
import sys
import argparse

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir,'..')
sys.path.insert(0,os.path.join(root_dir,'src'))

from tools.video import *
from tools.counter import get_counts,get_count_with_aff_matrix
from models.network_decoder import get_repnet_model
from config import config as cfg
np.set_printoptions(threshold=np.inf)

model = get_repnet_model('../weights')


imgs=np.load('../data/aff_matrix.npy')


pred_period, pred_score, within_period,per_frame_counts, chosen_stride = get_count_with_aff_matrix(
    model,
    imgs,
    stride=1,
    batch_size=20,
    threshold=cfg.THRESHOLD,
    within_period_threshold=cfg.WITHIN_PERIOD_THRESHOLD,
    constant_speed=cfg.CONSTANT_SPEED,
    median_filter=cfg.MEDIAN_FILTER,
    fully_periodic=cfg.FULLY_PERIODIC)

imgs, vid_fps = read_video('../data/test.mp4')

viz_reps(imgs, per_frame_counts, pred_score, interval=1000/cfg.VIZ_FPS,
        plot_score=cfg.PLOT_SCORE)

