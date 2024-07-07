from ultralytics import YOLO

def train(device: str, epoch:int = 100):
    model = YOLO('yolov8x-cls.pt')
    results = model.train(data='./dataset', epochs=epoch, imgsz=48, device=device)
    metric = int(results.top1 * 100)
    print(f"End of training with a top1 metric of {metric}")
    return metric