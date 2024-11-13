import base64
from io import BytesIO
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
from flask_socketio import emit
import numpy as np
from pyzbar.pyzbar import decode
from pathlib import Path
import sys
from PIL import Image
from flask import Flask, render_template, request, jsonify
from sevice.barcodedetection import detect_and_decode_barcode, save_and_display, enhance_barcode_image
from dotenv import load_dotenv, dotenv_values
import socket

app = Flask(__name__)
hostname = socket.gethostname()
pathenv = Path('./.env')
local_ip = socket.gethostbyname(hostname)
load_dotenv(dotenv_path=pathenv)
config = dotenv_values() 
app.config['SECRET_KEY'] = config.get('SECRET_KEY') or 'you-will-never-guess'
FONTEND_URL = config.get('FONTEND_URL') or 'http://localhost:3000'
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5000", "http://localhost:5000", "http://localhost:80", FONTEND_URL, f"http://{local_ip}"]}})
socketio = SocketIO(app, cors_allowed_origins=["http://127.0.0.1:5000", "http://localhost:5000", "http://localhost:80", FONTEND_URL, f"http://{local_ip}"])

@app.route("/")
def hello_world():
    return render_template("index.html")

def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

@socketio.on('frame')
def handle_frame(data):
    try:
        # รับข้อมูลรูปภาพจาก Frontend
        image_data = data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image_np = np.array(image)

        # แปลงจาก RGB เป็น BGR
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # ปรับปรุงรูปภาพและตรวจจับ Barcode
        enhanced_image = enhance_barcode_image(image_np)
        result_image, decoded_objects = detect_and_decode_barcode(enhanced_image)

        # แปลงรูปภาพเป็น Base64
        original_base64 = image_to_base64(image_np)
        processed_base64 = image_to_base64(result_image)

        # ข้อมูล Barcode
        barcodes = [{"type": obj.type, "data": obj.data.decode('utf-8')} for obj in decoded_objects]

        # ส่งผลลัพธ์กลับไปที่ Frontend
        emit('barcode_detected', {
            'barcodes': barcodes,
            'original_image': original_base64,
            'processed_image': processed_base64
        })
    except Exception as e:
        emit('error', {'message': str(e)})

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)
