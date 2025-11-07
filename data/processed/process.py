import os 
import shutil
import random

data_path ="H:\python\Driver_drowsiness_mlops\data\processed"

data_dir = r"H:\python\Driver_drowsiness_mlops\data\raw\drowsiness-detection"
train_dir = r"H:\python\Driver_drowsiness_mlops\data\processed\train"
val_dir = r"H:\python\Driver_drowsiness_mlops\data\processed\val"
classes =["closed_eye","open_eye"]


# Create folders
for split in [train_dir, val_dir]:
    for cls in classes:
        os.makedirs(os.path.join(split, cls), exist_ok=True)

# Split ratio
split_ratio = 0.8

# Move images
for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)
    
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    for img in train_images:
        shutil.copy(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
    for img in val_images:
        shutil.copy(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))

print("Data split done!")