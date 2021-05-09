import os
import argparse
from src.yolotf.inference import *
LOGGER = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser('A demo of using YOLOv5 in Tensorflow')
    parser.add_argument('--model-path', type=str,
                        default='yolo5s-tf/', required=True)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--outdir', type=str, default='output/')
    parser.add_argument('--video-path', type=str, default='demo.mp4')
    parser.add_argument('--capture-rate', type=int, default=10)
    parser.add_argument('--webcam', action='store_true')
    parser.add_argument('--classes', nargs='+')
    return parser.parse_args()


def main(args):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    vpath = 0 if args.webcam else args.video_path
    predictor = VideoInferenceNoShow(
        model_path=args.model_path,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        target_classes=args.classes
    )

    LOGGER.info("Starting the demo ...")
    LOGGER.info("Press 'q' to stop recording")
    predictor.process_video(
        vpath, capture_rate=args.capture_rate, outdir=args.outdir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
