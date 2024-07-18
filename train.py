from ultralytics import YOLO


model = YOLO("yolov8n.yaml")


# treinamento do modelo para prescrições
results = model.train(data="config.yaml", epochs=600)
