from ultralytics import YOLO
def main():
    model = YOLO('Weight/weights/best.pt')
    metrics = model.val(
        data='Dataset/data.yaml',
        split='test',
        imgsz=640,
        batch=32,
        plots=True,
    )
if __name__ == '__main__':
    main()