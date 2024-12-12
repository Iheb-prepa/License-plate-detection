from ultralytics import YOLO


model = YOLO("yolo11n.pt")

# results = model.train(data="license_plates_v11/data.yaml", epochs=1500)

# model = YOLO("runs/detect/train4/weights/last.pt") # Useful if you want to resume training

results = model.train(data="License_Plate_Recognition_10k/data.yaml", epochs=750)