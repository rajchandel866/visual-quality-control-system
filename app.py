from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from bottle_detection_pipeline import BottleDetectionPipeline
from edge_detection_pipeline import BottleEdgeDetectionPipeline
from defect_detection_pipeline import DefectDetectionPipeline, BottleDetector
import os
import cv2
import base64
import numpy as np

app = Flask(__name__)

VIDEO_PATH = './data/bottle_production.mp4'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/bottle_scanning', methods=['GET', 'POST'])
def bottle_scanning():
    if request.method == 'POST':
        blur_choice = request.form.get('blur_choice')
        blurRestOfFrame = True if blur_choice == '1' else False
        return Response(BottleEdgeDetectionPipeline().run_bottle_edge_detection_pipeline(VIDEO_PATH, blurRestOfFrame),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return render_template('bottle_scanning.html')

@app.route('/bottle_detection')
def bottle_detection():
    return Response(BottleDetectionPipeline().run_bottle_detection_pipeline(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/defect_detection')
def defect_detection():
    return render_template('defect_detection.html')

@app.route('/play_video')
def play_video():
    return render_template('play_video.html')

@app.route('/batch_process_folder', methods=['POST'])
def batch_process_folder():
    if 'folder_path' not in request.form:
        return jsonify({'message': 'Missing required fields: folder_path'}), 400

    folder_path = request.form['folder_path']
    template_path = './data/template.jpeg'

    output_directory = "output_directory"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    results = []
    for filename in os.listdir(folder_path):
        if not os.path.isdir(filename) and filename.lower().endswith('.png'):
            print(filename)
            image_path = os.path.join(folder_path, filename)
            result = DefectDetectionPipeline().template_matching(template_path, image_path, filename)
            results.append(result)

    return jsonify({'results': results})

# ✅ Final working live stream route
@app.route('/live_detection')
def live_detection():
    def generate():
        cap = cv2.VideoCapture(0)
        pipeline = DefectDetectionPipeline()

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.resize(frame, (640, 480))

            processed = pipeline.detect_defects_live(frame) if hasattr(pipeline, 'detect_defects_live') else frame

            _, buffer = cv2.imencode('.jpg', processed)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ✅ Batch defect detection with images encoded
@app.route('/batch_process_folder_base64', methods=['POST'])
def batch_process_folder_base64():
    folder_path = request.form.get('folder_path')

    if not folder_path or not os.path.exists(folder_path):
        return jsonify({'message': 'Invalid folder path'}), 400

    detector = BottleDetector("bottle_template.jpg")
    results = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)

            if img is None:
                continue

            processed_frame, match, top_left, bottom_right = detector.detect_defect_batch(img)

            _, buffer = cv2.imencode('.jpg', processed_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            results.append({
                'image_name': filename,
                'image': img_base64,
                'match': match,
                'top_left': top_left,
                'bottom_right': bottom_right
            })

    return jsonify({'results': results})

# Just an example route
@app.route("/index")
def index():
    return render_template("index.html")

# ✅ RUN THE APP
if __name__ == "__main__":
    app.run(debug=True)
