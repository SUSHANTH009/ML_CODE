import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import time
import queue
import threading
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import requests
import cv2
import io


app = Flask(__name__)
CORS(app)


data_queue = queue.Queue()


ESRGAN_MODEL_PATH = './esrgan_model'
YOLO_MODEL_PATH = 'best.pt'


yolo_model = YOLO(YOLO_MODEL_PATH)
esrgan_model = tf.saved_model.load(ESRGAN_MODEL_PATH)

logging.basicConfig(filename='pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/upload_blob', methods=['POST'])
def upload_blob():
    """
    Endpoint to handle blob file upload and save each frame with a unique filename in the queue.
    """
    app.logger.debug("Received POST request on /upload_blob")

    if 'video' not in request.files:
        app.logger.error("No video file in the request")
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')

    if not latitude or not longitude:
        app.logger.error("Missing latitude or longitude in the request")
        return jsonify({"error": "Missing latitude or longitude"}), 400

    timestamp = int(time.time() * 1000)
    filename = secure_filename(f"{timestamp}_{video_file.filename}")

    video_data = video_file.read()

    metadata = {
        "latitude": latitude,
        "longitude": longitude,
        "filename": filename,
        "timestamp": timestamp
    }


    data_queue.put({"video": video_data, "metadata": metadata})
    app.logger.info(f"Data added to queue: filename={filename}, metadata={metadata}")

    return jsonify({"message": "Frame and metadata added to queue successfully"}), 200


def preprocess_image(image):
    hr_image = tf.convert_to_tensor(image, dtype=tf.float32)
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    return tf.expand_dims(hr_image, 0)



def process_videos_from_queue():
    while True:

        if data_queue.empty():
            continue

        data_item = data_queue.get()
        video_data = data_item['video']
        metadata = data_item['metadata']
        latitude = metadata['latitude']
        longitude = metadata['longitude']
        filename = metadata['filename']

        try:
            video_stream = io.BytesIO(video_data)
            video_stream.seek(0)

            cap = cv2.VideoCapture(video_stream.read())
            pothole_detected = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break


                try:
                    lr_image = preprocess_image(frame)
                    sr_image = esrgan_model(lr_image)
                    sr_image = tf.squeeze(sr_image).numpy().astype(np.uint8)
                    logging.info(f"Frame enhanced from video {filename}.")

                except Exception as e:
                    logging.error(f"Error enhancing frame from video {filename}: {str(e)}")
                    continue


                try:
                    results = yolo_model.predict(sr_image)

                    for r in results:
                        if len(r.boxes) > 0:
                            pothole_detected = True
                            break

                    if pothole_detected:
                        break  

                except Exception as e:
                    logging.error(f"Error during YOLO detection on video {filename}: {str(e)}")

            cap.release()

            if pothole_detected:
                url = "http://localhost:8080/api/location/"
                data = {
                    "latitude": latitude,
                    "longitude": longitude
                }
                response = requests.post(url, json=data)
                if response.status_code == 200:
                    logging.info(f"Pothole detected in video {filename}, GPS location sent successfully.")
                else:
                    logging.error(f"Failed to send GPS location for video {filename}. Response: {response.text}")

            else:
                logging.info(f"No pothole detected in video {filename}. Skipping GPS notification.")

        except Exception as e:
            logging.error(f"Error processing video {filename}: {str(e)}")



def start_flask():
    app.run(debug=True, use_reloader=False)



def start_processing():
    processing_thread = threading.Thread(target=process_videos_from_queue, daemon=True)
    processing_thread.start()


if __name__ == '__main__':
    threading.Thread(target=start_flask, daemon=True).start()

    start_processing()

    while True:
        time.sleep(1)