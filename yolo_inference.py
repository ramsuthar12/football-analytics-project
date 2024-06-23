from ultralytics import YOLO # type: ignore

model = YOLO('models/best.pt')

results = model.predict('input_videos\B1606b0e6_1 (36).mp4', save = True)

print(results[0])
print('----------------------------------------------------')

for box in results[0].boxes:
    print(box)