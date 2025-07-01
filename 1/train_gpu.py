from ultralytics import YOLO

if __name__ == '__main__':
    # 加载预训练的 YOLOv8 模型
    model = YOLO('yolov8m.pt')

    # 使用 GPU 训练
    model.train(
        data="D:\yolov8_all\yolo11\climb_railings_yolo11\data.yaml",  # 数据集配置文件
        imgsz=640,         # 输入图像尺寸
        epochs=50,         # 训练轮次
        batch=8,           # GPU 批次大小可以适当增大
        device=0,          # 指定使用 GPU 设备 ID（0 表示第一块 GPU）
        name="yolov8m_gpu" # 训练任务名称
    )