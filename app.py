#this is group face recognition system 
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import csv
import os
import warnings
import time
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

load_dotenv(override=True)
secret_key = os.getenv("SECRET_KEY")
app = Flask(__name__)

app.secret_key = secret_key

# -----------------------------
# TEACHER LOGIN
# -----------------------------
ADMIN_USERNAME = "teacher"
ADMIN_PASSWORD = "12345"

# -----------------------------
# LOAD FACE DATABASE
# -----------------------------
db_path = "face_db.pkl"
if os.path.exists(db_path):
    with open(db_path, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

for k, v in face_db.items():
    if isinstance(v, np.ndarray):
        face_db[k] = [v]

# -----------------------------
# INIT MODEL
# -----------------------------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1)

cap = None
camera_on = False
THRESHOLD = 0.65
attendance_today = set()

today = datetime.now().strftime("%Y-%m-%d")
attendance_file = f"attendance_{today}.csv"

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "DateTime"])

# -----------------------------
# LOGIN REQUIRED DECORATOR
# -----------------------------
def login_required(func):
    def wrapper(*args, **kwargs):
        if "admin" not in session:
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

# -----------------------------
# VIDEO STREAM
# -----------------------------
def generate_frames():
    global cap, camera_on

    while True:
        if not camera_on or cap is None:
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            continue

        faces = face_app.get(frame)

        for face in faces:
            emb = face.embedding.reshape(1, -1)
            name = "Unknown"
            best_score = 0

            for db_name, db_embs in face_db.items():
                avg_emb = np.mean(db_embs, axis=0).reshape(1, -1)
                score = cosine_similarity(emb, avg_emb)[0][0]
                if score > best_score:
                    best_score = score
                    name = db_name

            if best_score < THRESHOLD:
                name = "Unknown"

            box = face.bbox.astype(int)
            color = (0,255,0) if name!="Unknown" else (0,0,255)

            cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),color,2)
            cv2.putText(frame,name,(box[0],box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

            if name!="Unknown" and name not in attendance_today:
                attendance_today.add(name)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(attendance_file,"a",newline="") as f:
                    csv.writer(f).writerow([name, now])

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame + b"\r\n")

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return redirect("/dashboard") if "admin" in session else redirect("/login")

# -----------------------------
# LOGIN
# -----------------------------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == ADMIN_USERNAME and request.form["password"] == ADMIN_PASSWORD:
            session["admin"] = ADMIN_USERNAME
            return redirect("/dashboard")
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# -----------------------------
# DASHBOARD
# -----------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/video")
@login_required
def video():
    return Response(generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

# -----------------------------
# CAMERA CONTROL
# -----------------------------
@app.route("/start_camera", methods=["POST"])
@login_required
def start_camera():
    global cap, camera_on
    print("START CAMERA CALLED")   

    if not camera_on:
        cap = cv2.VideoCapture(0)
        print("Camera object:", cap.isOpened())  
        camera_on = True

    return jsonify({"status":"Camera Started"})

@app.route("/stop_camera", methods=["POST"])
@login_required
def stop_camera():
    global cap, camera_on
    camera_on = False
    if cap:
        cap.release()
        cap = None
    return jsonify({"status":"Camera Stopped"})

# -----------------------------
# ADD USER
# -----------------------------
@app.route("/add_user", methods=["POST"])
@login_required
def add_user():
    global cap
    if cap is None:
        return jsonify({"message":"Start camera first"}), 400

    name = request.json.get("name")
    if not name:
        return jsonify({"message":"Name required"}), 400

    ret, frame = cap.read()
    if not ret:
        return jsonify({"message":"Camera error"}), 400

    faces = face_app.get(frame)
    if not faces:
        return jsonify({"message":"No face detected"}), 400

    face_db.setdefault(name, []).append(faces[0].embedding)

    with open(db_path,"wb") as f:
        pickle.dump(face_db,f)

    return jsonify({"message":"User added successfully"})

# -----------------------------
# DELETE USER
# -----------------------------
@app.route("/delete_user", methods=["POST"])
@login_required
def delete_user():
    name = request.json.get("name")
    if name in face_db:
        del face_db[name]
        with open(db_path,"wb") as f:
            pickle.dump(face_db,f)
        return jsonify({"message":"User deleted"})
    return jsonify({"message":"User not found"}), 404

# -----------------------------
# ATTENDANCE
# -----------------------------
@app.route("/attendance")
@login_required
def attendance():
    records = []
    if os.path.exists(attendance_file):
        with open(attendance_file,"r") as f:
            reader = csv.reader(f)
            next(reader)
            records = list(reader)
    return render_template("attendance.html", records=records)

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, threaded=True)