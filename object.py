import threading
import cv2
from flask import Flask, render_template_string, request, jsonify, Response

app = Flask(__name__)


config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)
classLabels = []
file_name = 'labels.txt'

with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127, 127.5))
model.setInputSwapRB(True)


def run_flask():
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)


flask_thread = threading.Thread(target=run_flask)
flask_thread.start()


@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Object Detection</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; text-align: center; }
            button { background-color: #4CAF50; color: white; padding: 10px 20px; font-size: 16px; margin: 10px; border: none; cursor: pointer; border-radius: 5px; }
            button:hover { background-color: #45a049; }
            .status { margin-top: 20px; font-size: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Object Detection</h1>
            <button onclick="startImageDetection()">Start Image Detection</button>
            <button onclick="startVideoDetection()">Start Video Detection</button>
            <button onclick="startWebcamDetection()">Start Webcam Detection</button>
            <button onclick="stopDetection()">Stop Detection</button>
            <div class="status" id="status"></div>
            <br>
            <img id="output" src="" style="max-width: 100%;"/>
        </div>
        <script>
            function startImageDetection() {
                fetch('/start_image_detection', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').innerText = data.status;
                        document.getElementById('output').src = '/output_image';
                    });
            }

            function startVideoDetection() {
                document.getElementById('output').src = '/video_feed';
                document.getElementById('status').innerText = 'Video Detection Started!';
            }

            function startWebcamDetection() {
                document.getElementById('output').src = '/webcam_feed';
                document.getElementById('status').innerText = 'Webcam Detection Started!';
            }

            function stopDetection() {
                fetch('/stop_detection', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').innerText = data.status;
                        document.getElementById('output').src = "";
                    });
            }
        </script>
    </body>
    </html>
    """

@app.route('/start_image_detection', methods=['POST'])
def start_image_detection():
    img = cv2.imread('tomato.jpg')
    ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.55)

    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        if ClassInd <= 80:
            cv2.rectangle(img, boxes, (255, 0, 0), 2)
            cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imwrite('output_image.png', img)
    return jsonify(status='Image Detection Complete')

@app.route('/output_image')
def output_image():
    return Response(open('output_image.png', 'rb').read(), mimetype='image/png') # add your image path

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        detection_result = model.detect(frame, confThreshold=0.55)
        if detection_result is not None and len(detection_result) == 3:
            ClassIndex, confidence, bbox = detection_result
            if len(ClassIndex) > 0:
                for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                    if ClassInd <= 80:
                        cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                        cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames('dataset/video.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame') # add your video path

@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame') 

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    cv2.destroyAllWindows()
    return jsonify(status='Detection Stopped')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)
