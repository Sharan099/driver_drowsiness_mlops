import os
import shutil
import time

def preprocess_data():
    """
    Move new images from Test_data to raw dataset with timestamped filenames
    """
    # -----------------------------
    # Source directories (new images)
    # -----------------------------
    test_base = r"H:\python\Driver_drowsiness_mlops\data\raw\drowsiness-detection\Test_data"
    test_open = os.path.join(test_base, "open_eye")
    test_closed = os.path.join(test_base, "closed_eye")

    # -----------------------------
    # Destination directories (main dataset)
    # -----------------------------
    raw_base = r"H:\python\Driver_drowsiness_mlops\data\raw\drowsiness-detection"
    raw_open = os.path.join(raw_base, "open_eye")
    raw_closed = os.path.join(raw_base, "closed_eye")

    os.makedirs(raw_open, exist_ok=True)
    os.makedirs(raw_closed, exist_ok=True)

    # -----------------------------
    # Process Open Eye Images
    # -----------------------------
    for filename in os.listdir(test_open):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            src_path = os.path.join(test_open, filename)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            dst_filename = f"open_{timestamp}.png"
            dst_path = os.path.join(raw_open, dst_filename)
            shutil.copy2(src_path, dst_path)
            print(f"✅ Copied Open Eye: {dst_filename}")

    # -----------------------------
    # Process Closed Eye Images
    # -----------------------------
    for filename in os.listdir(test_closed):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            src_path = os.path.join(test_closed, filename)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            dst_filename = f"closed_{timestamp}.png"
            dst_path = os.path.join(raw_closed, dst_filename)
            shutil.copy2(src_path, dst_path)
            print(f"✅ Copied Closed Eye: {dst_filename}")

    print("✅ Preprocessing complete! New images added to raw dataset.")

    #######
    # splitting the raw data into train and val files 
    #######

    data_path =r"H:\python\Driver_drowsiness_mlops\data\processed"
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
