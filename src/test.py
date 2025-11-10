import cv2
import torch
from torchvision import transforms
from PIL import Image
from cnn import SimpleCNN  # your CNN class
import os
import time

# -----------------------------
# 1️⃣ Load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(r"H:\python\Driver_drowsiness_mlops\Outputs\models\best_model.pth", map_location=device))
model.eval()

# -----------------------------
# 2️⃣ Transform for CNN input
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# 3️⃣ Load Haar cascades
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# -----------------------------
# 4️⃣ Setup directory to save drowsy frames
# -----------------------------
save_dir = r"H:\python\Driver_drowsiness_mlops\Outputs\drowsy_frames"
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# 5️⃣ Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)
print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    status = "😎 Awake"  # default status

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (fx, fy, fw, fh) in faces:
        face_roi = gray[fy:fy+fh, fx:fx+fw]

        # Detect eyes inside face
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)

        eye_predictions = []

        for (ex, ey, ew, eh) in eyes[:2]:  # track up to 2 eyes
            # Draw rectangle
            cv2.rectangle(frame, (fx+ex, fy+ey), (fx+ex+ew, fy+ey+eh), (0, 255, 0), 2)

            # Crop and preprocess eye
            eye_img = face_roi[ey:ey+eh, ex:ex+ew]
            eye_img_resized = cv2.resize(eye_img, (227, 227))
            eye_pil = Image.fromarray(eye_img_resized)
            input_tensor = transform(eye_pil).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                pred = (output > 0.5).float()
                eye_predictions.append(pred.item())

        # Even one eye closed → Drowsy
        if len(eye_predictions) > 0:
            if 0 in eye_predictions:  # 0 = drowsy
                status = "😴 Drowsy"

                # Save frame with timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(save_dir, f"drowsy_{timestamp}.png")
                cv2.imwrite(save_path, frame)
                print(f"⚠️ Drowsy detected! Frame saved: {save_path}")

    # Display status on frame
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if status == "😴 Drowsy" else (0, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)

    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
