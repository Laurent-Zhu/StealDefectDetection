# -*- coding: utf-8 -*-

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'runs/train/exp7/weights/best.pt')
    model.predict(source=r'detect_images/crazing_166.jpg',
                  save=True,
                  show=True,
                  )
