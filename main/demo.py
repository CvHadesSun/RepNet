
import os
import sys
import argparse

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir,'..')
sys.path.insert(0,os.path.join(root_dir,'src'))

from tools.video import *
from tools.counter import get_counts
from models.network import get_repnet_model
from config import config as cfg
# from models.network import ResnetPeriodEstimator


def parser_upate():
    parser = argparse.ArgumentParser(description='RepNet.')
    parser.add_argument('--ckpt' ,'-c', type=str, default='../weights',
                    help='checkpoints to load.')
    parser.add_argument('--video','-v' , type=str, default='../data/test.mp4',
                    help='the video to test.')
    parser.add_argument('--output','-o',  type=str, default='../data/',
                    help='the test video output dir.')  
    return parser.parse_args()



if __name__=="__main__":
    args=parser_upate()
    print(args)
    # 
    # need VPN
    if not os.path.exists(args.video):
        print("Downloading video from {}...".format(cfg.VIDEO_URL))
        video_name=cfg.VIDEO_URL.split('/')[-1]+'.mp4'

        download_video_from_url(cfg.VIDEO_URL,
                            path_to_video=os.path.join('../data',video_name))
        
        args.video=os.path.join('../data',video_name)
    imgs, vid_fps = read_video(args.video)
    print("<<images number:{}".format(len(imgs)),"="*5,">>fps:{}".format(vid_fps))


    # get model 

    model = get_repnet_model(args.ckpt)
    print('Running RepNet...') 
    model.summary()

    pred_period, pred_score, within_period,per_frame_counts, chosen_stride = get_counts(
        model,
        imgs,
        strides=[1,2,3,4],
        batch_size=20,
        threshold=cfg.THRESHOLD,
        within_period_threshold=cfg.WITHIN_PERIOD_THRESHOLD,
        constant_speed=cfg.CONSTANT_SPEED,
        median_filter=cfg.MEDIAN_FILTER,
        fully_periodic=cfg.FULLY_PERIODIC)

    print('Visualizing results...') 
    viz_reps(imgs, per_frame_counts, pred_score, interval=1000/cfg.VIZ_FPS,
         plot_score=cfg.PLOT_SCORE)
    
        

    


# runing repnet

# (pred_period, pred_score, within_period,

# print('Visualizing results...') 
# viz_reps(imgs, per_frame_counts, pred_score, interval=1000/VIZ_FPS,
#          plot_score=PLOT_SCORE)


# # Debugging video showing scores, per-frame frequency prediction and 
# # within_period scores.
# create_count_video(imgs,
#                    per_frame_counts,
#                    within_period,
#                    score=pred_score,
#                    fps=vid_fps,
#                    output_file='/tmp/debug_video.mp4',
#                    delay=1000/VIZ_FPS,
#                    plot_count=True,
#                    plot_within_period=True,
#                    plot_score=True)
# show_video('/tmp/debug_video.mp4')
