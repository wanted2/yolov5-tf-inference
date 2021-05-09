YOLOv5 in Tensorflow
=====

This is a re-implementation of [Pytorch's YOLO v5](https://github.com/ultralytics/yolov5) in Tensorflow 2.4.x.
There is **no training functions**, just some inference functions and a model zoo.

## Usage

### Requirements

* Python >= 3.8.5
* Tensorflow >=2.4.1

Others in [`requirements.txt`](./requirements.txt).

### Usage

```
$ python demo.py --model-path yolo5s-tf/ --conf-thres 0.25 --iou-thres 0.45 --classes 0 1 2 \
    --outdir output/ --capture-rate 10 --webcam --video-path ./demo.mp4
```

### Model Zoo

| Model name | Pytorch | Tensorflow (this repo) | No. params |
| --- | --- | --- | --- |
| yolov5s | [model](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt) | [model](https://drive.google.com/file/d/1AYEwzkwe8g5MQdMONvcAw66to8q8ghgW/view?usp=sharing)| 7M |

More models will be added soon.
Some evaluations to compare the models in a benchmark will be added.