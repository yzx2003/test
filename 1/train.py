from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'yolov11.yaml')  # 此处以 m 为例，只需写yolov11m即可定位到m模型
    model.train(
        data="data.yaml",  # 自定义数据集配置
        imgsz=640,  # 输入图像尺寸
        epochs=50,  # 训练轮次
        batch=8,  # CPU批次大小
        device="cpu",  # CPU训练
        name="yolov11m",
    )