# app.py
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
from sevice.barcodedetection import detect_and_decode_barcode, save_and_display
from dotenv import load_dotenv, dotenv_values
import socket


app = Flask(__name__)
hostname = socket.gethostname()
pathenv = Path('./.env')
local_ip = socket.gethostbyname(hostname)
load_dotenv(dotenv_path=pathenv)
config = dotenv_values() 
app.config['SECRET_KEY'] = config.get('SECRET_KEY') or 'you-will-never-give-you-up'
FONTEND_URL = config.get('FONTEND_URL') or 'http://localhost:3000'
# Allow CORS for all routes in Flask
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5000", "http://localhost:5000", "http://localhost:80",FONTEND_URL,f"http://{local_ip}" ]}})
# Set accepted origins explicitly for SocketIO,f"http://{local_ip}"
socketio = SocketIO(app, cors_allowed_origins=["http://127.0.0.1:5000", "http://localhost:5000", "http://localhost:80",FONTEND_URL,f"http://{local_ip}"])

@app.route("/")
def hello_world():
    return render_template("index.html")    

def loadImage():
    # Load image
    image_path = "./public/assets/testbarcode.png"
    
    # Try multiple image loading methods
    image = None
    try:
        # Try PIL first
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except:
        # Fallback to OpenCV
        image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    try:
        # Process image
        result_image, decoded_objects = detect_and_decode_barcode(image)
        
        if not decoded_objects:
            print("\nNo barcodes detected. Troubleshooting info:")
            print("1. Enhanced image saved as 'enhanced_barcode.png'")
            print("2. Check if image path is correct:", image_path)
            print("3. Image dimensions:", image.shape)
            print("4. Trying alternative display method...")
        
        # Save and display result
        save_and_display(result_image)
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print("Stack trace:", sys.exc_info())


def main():
    select = input("Select an option: \n1. Load image from file\n2. Use webcam\n")
    if select == "1":
        loadImage()
    elif select == "2":
        webCam()
    
    
def webCam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Process the frame for barcode detection
        detected_frame, decoded_objects = detect_and_decode_barcode(frame)
        # Display the result
        cv2.imshow("Barcode Detection", detected_frame)
        if decoded_objects :
            save_and_display(detected_frame)
            break        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


@app.route('/scan-barcode', methods=['POST'])
def scan_barcode():
    # Get the image from the POST request
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No image file provided"}), 400
    
    # Convert the file to a PIL image
    pil_image = Image.open(file)
    image = np.array(pil_image)
    
    # If the image is in RGB, convert it to BGR for OpenCV processing
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Use your function to detect barcodes
    result_image, decoded_objects = detect_and_decode_barcode(image)
    
    # Format barcode data as JSON
    barcodes = [{"type": obj.type, "data": obj.data.decode('utf-8')} for obj in decoded_objects]
    
    return jsonify({"barcodes": barcodes})

@socketio.on('frame')
def handle_frame(data):
    try:
        # Convert the received image data to a PIL image
        image_data = data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image_np = np.array(image)

        # Convert RGB to BGR if needed
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Detect barcodes
        original, enhanced, decoded_objects = detect_and_decode_barcode(image_np, debug=True)

        # Convert images back to base64
        _, buffer_original = cv2.imencode('.jpg', original)
        _, buffer_enhanced = cv2.imencode('.jpg', enhanced)
        
        original_image = base64.b64encode(buffer_original).decode('utf-8')
        processed_image = base64.b64encode(buffer_enhanced).decode('utf-8')

        # Return results
        barcodes = [{"type": obj.type, "data": obj.data.decode('utf-8')} for obj in decoded_objects]
        if barcodes:
            emit('barcode_detected', {
                'barcodes': barcodes,
                'original_image': f'data:image/jpeg;base64,{original_image}',
                'processed_image': f'data:image/jpeg;base64,{processed_image}'
            })
        else:
            emit('barcode_not_detected', {
                'message': 'ไม่พบบาร์โค้ดในภาพ กรุณาลองใหม่อีกครั้ง',
                'original_image': f'data:image/jpeg;base64,{original_image}',
                'processed_image': f'data:image/jpeg;base64,{processed_image}'
            })
            
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        emit('error', {'message': f'เกิดข้อผิดพลาด: {str(e)}'})

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)