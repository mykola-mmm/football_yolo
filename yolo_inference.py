from ultralytics import YOLO

# model = YOLO('yolov8n')
model = YOLO('yolov8x')

# results = model.predict("./train_vids/08fd33_4.mp4", save=True, save_path="./output_vids/08fd33_4.mp4")
results = model.predict("./train_vids/08fd33_4.mp4", save=True)
print(results[0])
print("+++++++++++++++++++++++===")

for box in results[0].boxes:
    print(box)