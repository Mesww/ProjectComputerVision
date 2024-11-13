# barcodedetection.py
from typing import Tuple
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import os
import sys
from PIL import Image

def enhance_barcode_image(image, debug=False):
    """
    Enhance image for better barcode detection
    
    Args:
        image: Input image (BGR format)
        debug: If True, returns debug information
    
    Returns:
        Enhanced image and debug images dictionary if debug=True
    """
    debug_images = {}
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        debug_images['1_grayscale'] = gray.copy()
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 51, 9
    )
    if debug:
        debug_images['2_threshold'] = thresh.copy()
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    if debug:
        debug_images['3_denoised'] = denoised.copy()
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    if debug:
        debug_images['4_enhanced'] = enhanced.copy()
    
    return enhanced, debug_images if debug else None

def detect_and_decode_barcode(image, debug=False):
    """
    Detect and decode barcodes using multiple processing methods
    
    Args:
        image: Input image (BGR format)
        debug: If True, returns debug information
    
    Returns:
        Tuple of (original image, enhanced image, decoded objects)
    """
    # Keep a copy of original image
    original_image = image.copy()
    decoded_objects = []
    
    # Convert to grayscale if image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Try different processing methods
    processing_methods = [
        lambda img: img,  # Original
        lambda img: cv2.GaussianBlur(img, (5, 5), 0),  # Blur
        lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # Otsu
        lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),  # Adaptive
    ]
    
    enhanced = None
    for process in processing_methods:
        processed = process(gray)
        
        # Try different scaling factors
        for scale in [1.0, 1.5, 2.0]:
            if scale != 1.0:
                width = int(processed.shape[1] * scale)
                height = int(processed.shape[0] * scale)
                dim = (width, height)
                scaled = cv2.resize(processed, dim, interpolation=cv2.INTER_CUBIC)
            else:
                scaled = processed
            
            # Try normal and inverted
            for img in [scaled, cv2.bitwise_not(scaled)]:
                decoded = decode(img)
                if decoded:
                    decoded_objects = decoded
                    enhanced = processed
                    break
            if decoded_objects:
                break
        if decoded_objects:
            break
    
    # If still no detection, try advanced enhancement
    if not decoded_objects:
        enhanced, debug_images = enhance_barcode_image(image, debug=debug)
        if debug and debug_images:
            for step_name, img in debug_images.items():
                cv2.imwrite(f"{step_name}.png", img)
        
        decoded_objects = decode(enhanced)
    
    # Draw detected barcodes if any found
    if decoded_objects:
        for obj in decoded_objects:
            # Draw boundary
            points = obj.polygon
            if points is None:
                continue
                
            if len(points) > 4:
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                points = [tuple(point[0]) for point in hull]
            
            n = len(points)
            for j in range(n):
                cv2.line(original_image,
                        (int(points[j][0]), int(points[j][1])),
                        (int(points[(j+1) % n][0]), int(points[(j+1) % n][1])),
                        (0, 255, 0), 2)
                
                if enhanced is not None:
                    cv2.line(enhanced,
                            (int(points[j][0]), int(points[j][1])),
                            (int(points[(j+1) % n][0]), int(points[(j+1) % n][1])),
                            (0, 255, 0), 2)
            
            # Add text annotation
            barcode_data = obj.data.decode('utf-8')
            barcode_type = obj.type
            x, y = points[0]
            cv2.putText(original_image, f"{barcode_data} ({barcode_type})",
                      (int(x), int(y) - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                      
            if enhanced is not None:
                cv2.putText(enhanced, f"{barcode_data} ({barcode_type})",
                          (int(x), int(y) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                          
            print(f"Found {barcode_type} barcode: {barcode_data}")
    else:
        print("No barcodes detected in the image")
        if enhanced is None:
            enhanced = image.copy()
            
    return original_image, enhanced, decoded_objects
def save_and_display(image, title="Processed Image"):
    """
    Save image and attempt to display using PIL
    """
    user_input = input("Do you want to save the image? (Y/n): ")
    user_input.strip()
    user_input.lower()
                
    try:
        if user_input == "":
            user_input = "y"
        if user_input == "y" or user_input == "yes":
            print(f"Saving image... {user_input}") 
            # Save the image
            output_path = "./public/assets/processed_barcode.png"
            cv2.imwrite(output_path, image)
            print(f"Saved processed image to: {output_path}")
    except user_input != "y"  or user_input != "n"  or user_input != "" or user_input != "yes" or user_input != "no" :
        print("Invalid input. Please enter 'y' or 'n'.")
        return
    except Exception as e:
        print(f"Could not save image: {e}")
    
    # Try to display using PIL
    try:
        # Convert from BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        pil_image.show()
    except Exception as e:
        print(f"Could not display image: {e}")
        print("Image has been saved to disk instead.")

