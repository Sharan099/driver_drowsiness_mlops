from flask import Flask, render_template, Response
import cv2
import torch
from torchvision import transforms
from PIL import Image
import os, time, sys
import threading
import winsound  # For beep

sys.path.append(r"H:\python\Driver_drowsiness_mlops\src")
from cnn import SimpleCNN

app = Flask(__name__)

# -----------------------------
# 1️⃣ Load CNN model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(
    r"H:\python\Driver_drowsiness_mlops\Outputs\models\best_model.pth", map_location=device))
model.eval()

# -----------------------------
# 2️⃣ Transform for CNN input
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# 3️⃣ Haar cascades
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# -----------------------------
# 4️⃣ Save drowsy frames
# -----------------------------
save_dir = r"H:\python\Driver_drowsiness_mlops\Outputs\drowsy_frames"
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# 5️⃣ Beep alert (threaded)
# -----------------------------
def beep_alert():
    try:
        winsound.Beep(1000, 300)  # 1000 Hz for 300ms
    except RuntimeError:
        # Ignore beep errors to avoid crashing the stream
        pass

# -----------------------------
# 6️⃣ Frame generator for streaming
# -----------------------------
def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        status = "😎 Awake"

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (fx, fy, fw, fh) in faces:
            face_roi_gray = gray[fy:fy+fh, fx:fx+fw]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5)
            eye_predictions = []

            # CNN prediction for each eye
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_crop = face_roi_gray[ey:ey+eh, ex:ex+ew]
                eye_resized = cv2.resize(eye_crop, (227, 227))
                eye_tensor = transform(Image.fromarray(eye_resized)).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = (model(eye_tensor) > 0.5).float().item()  # 1 = awake, 0 = drowsy
                    eye_predictions.append(pred)

            # Decide status
            if len(eye_predictions) > 0 and 0 in eye_predictions:
                status = "😴 Drowsy"
                # Save frame
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(os.path.join(save_dir, f"drowsy_{timestamp}.png"), frame)
                # Beep in a separate thread
                threading.Thread(target=beep_alert, daemon=True).start()
            else:
                status = "😎 Awake"

            # Draw eye rectangles
            for (ex, ey, ew, eh) in eyes[:2]:
                color = (0,0,255) if status=="😴 Drowsy" else (0,255,0)
                cv2.rectangle(frame[fy:fy+fh, fx:fx+fw], (ex, ey), (ex+ew, ey+eh), color, 2)

        # Overlay status
        color = (0,0,255) if status=="😴 Drowsy" else (0,255,0)
        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# -----------------------------
# 7️⃣ Flask routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------
# 8️⃣ Run Flask app
# -----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
