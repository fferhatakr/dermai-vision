import os
import cv2
from ultralytics import YOLO

model = YOLO('models/detector/best.pt')
input_folder = 'data/processed/kfold_data'
output_base = 'data/processed/just_leke'


for class_name in os.listdir(input_folder):
    class_input_dir = os.path.join(input_folder,class_name)
    class_output_dir = os.path.join(output_base,class_name)


    if os.path.isdir(class_input_dir):

        os.makedirs(class_output_dir,exist_ok=True)
        print({class_name})

        for img_name in os.listdir(class_input_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):

                img_path = os.path.join(class_input_dir,img_name)

                results = model.predict(img_path, conf=0.5, verbose=False)


                for r in results:
                    if len(r.boxes) > 0:

                        b= r.boxes.xyxy[0].cpu().numpy().astype(int)

                        img = cv2.imread(img_path)
                        crop = img[b[1]:b[3], b[0]:b[2]]
                        cv2.imwrite(os.path.join(class_output_dir, img_name), crop)

print("Success")
