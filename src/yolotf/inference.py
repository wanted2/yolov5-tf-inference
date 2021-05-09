from typing import Any, Iterable
import tensorflow as tf
import numpy as np
from src.tools.common import Timer, setup_logger, Paralellism
from src.yolotf.utils import load_image, load_yolo_model, non_max_suppression
import cv2
from PIL import Image, ImageDraw
LOGGER = setup_logger(__name__)


class Inference:
    m = None

    def __init__(self, model_path: str) -> None:
        try:
            self.m = load_yolo_model(model_path)[0]
        except Exception as e:
            LOGGER.error(e)

    def process(self, x: Any) -> Any:
        raise NotImplementedError('This should be implemented in children')


class ImageInference(Inference):
    def __init__(self, model_path: str,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 target_classes: Iterable[str] = None) -> None:
        super().__init__(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = target_classes

    @Timer.measure_runtime
    def process(self, x: Any) -> Any:
        if isinstance(x, str):
            x = load_image(image_path=x, target_size=640)[0]
        elif isinstance(x, tf.Tensor):
            x = x.numpy()
        if len(x.shape) != 4:
            x = x[np.newaxis, ...]
        pred = self.m(images=x)
        pred = pred['output_0']
        filtered_results = non_max_suppression(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=self.classes,
        )[0]
        return filtered_results


class VideoInference(ImageInference):
    def __init__(self, model_path: str,
                 conf_thres: float,
                 iou_thres: float,
                 target_classes: Iterable[str]) -> None:
        super().__init__(model_path, conf_thres=conf_thres,
                         iou_thres=iou_thres, target_classes=target_classes)

    def process(self, x):
        results = super().process(x)[0]
        return results

    def process_video(self, video_target: str or int, capture_rate: int = 300):
        cap = cv2.VideoCapture(video_target)
        if not cap.isOpened():
            raise ValueError(f'Cannot open {video_target}')
        count = 0
        while True:
            if count % capture_rate == 0:
                ret, img = cap.read()
                if not ret or img is None:
                    break
                x = img.copy()[:, :, [2, 1, 0]]
                x = cv2.resize(x, (640, 640)).astype('float32')
                x = x.transpose([-1, 0, 1])
                results = self.process(x / 255.)
                vis_img = Image.fromarray(img)  # img.copy()
                draw = ImageDraw.Draw(vis_img)
                H, W, _ = img.shape
                for bbox in results[0]:
                    x1, y1, x2, y2, _, _ = bbox
                    x1, y1, x2, y2 = max(0, x1/640 * W), max(0, y1 / 640 * H), \
                        x2 / 640 * W, y2/640 * H
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    draw.rectangle((x1, y1, x2, y2), outline='green', width=3)
                cv2.imshow('vis', np.array(vis_img))
                if cv2.waitKey(30) == ord('q'):
                    break
            else:
                cap.grab()
            count += 1
        cap.release()
        cv2.destroyAllWindows()


class VideoInferenceNoShow(VideoInference):
    def __init__(self, model_path: str, conf_thres: float,
                 iou_thres: float, target_classes: Iterable[str]) -> None:
        super().__init__(model_path, conf_thres, iou_thres, target_classes)

    @Paralellism.threadize
    def process(self, x, img, out: str):
        results = super().process(x)
        np.savez_compressed(out, data=results[0])
        LOGGER.debug(f'Output numpy array to {out}')
        vis_img = img.copy()
        H, W, _ = img.shape
        for bbox in results[0]:
            x1, y1, x2, y2, _, _ = bbox
            x1, y1, x2, y2 = x1 * W / 640, y1 / H * 640, \
                x2 * W / 640, y2 / H * 640
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2),
                          (0, 255, 0), 3, cv2.LINE_8)
        cv2.imwrite(f'{out}.jpg', vis_img)

    def process_video(self, video_target: str or int, capture_rate: int = 300, outdir: str = 'data'):
        cap = cv2.VideoCapture(video_target)
        if not cap.isOpened():
            raise ValueError(f'Cannot open {video_target}')
        count = 0
        while True:
            if count % capture_rate == 0:
                ret, img = cap.read()
                if not ret or img is None:
                    break
                x = img.copy()[:, :, [2, 1, 0]]
                x = cv2.resize(x, (640, 640)).astype('float32')
                x = x.transpose([-1, 0, 1])
                self.process(x / 255., img, out=f'{outdir}/{count}')
                cv2.imshow('vis', np.array(img))
                if cv2.waitKey(30) == ord('q'):
                    break
            else:
                cap.grab()
            count += 1
        cap.release()
        cv2.destroyAllWindows()
