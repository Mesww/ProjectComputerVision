# barcodedetection.py
from typing import Tuple
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import os
import sys
from PIL import Image
from skimage import exposure
from skimage.measure import label, regionprops
from skimage.feature import match_template


# Frequency domain filtering function
def frequency_domain_filtering(image, cutoff=30):
    img_float = np.float32(image)
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = cutoff
    center = [crow, ccol]
    cv2.circle(mask, center, r, 0, -1)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return np.uint8(img_back)

# Sharpening function
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  # Sharpening kernel
    return cv2.filter2D(image, -1, kernel)

# Histogram matching function
def histogram_matching(image, reference_image):
    return exposure.match_histograms(image, reference_image, multichannel=False)

# Edge detection function
def edge_detection(image, low_threshold=100, high_threshold=200):
    return cv2.Canny(image, low_threshold, high_threshold)

# Morphological operations function
def morphological_operations(image, operation='dilation'):
    kernel = np.ones((5, 5), np.uint8)
    if operation == 'dilation':
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == 'erosion':
        return cv2.erode(image, kernel, iterations=1)
    elif operation == 'opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Function to apply interpolation
def apply_interpolation(image, scale=1.0, interpolation=cv2.INTER_CUBIC):
    height, width = image.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    dim = (new_width, new_height)
    resized_image = cv2.resize(image, dim, interpolation=interpolation)
    return resized_image

# Geometric transformation functions (e.g., rotation, scaling, affine)
def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def affine_transform(image, matrix):
    height, width = image.shape[:2]
    transformed_image = cv2.warpAffine(image, matrix, (width, height))
    return transformed_image

# Image enhancement for barcode detection
def enhance_barcode_image(image, reference_image=None, debug=False):
    debug_images = {}
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        debug_images['1_grayscale'] = gray.copy()
    
    # Apply frequency domain filtering
    freq_filtered = frequency_domain_filtering(gray)
    if debug:
        debug_images['2_freq_filtered'] = freq_filtered.copy()
    
    # Apply sharpening
    sharpened = sharpen_image(freq_filtered)
    if debug:
        debug_images['3_sharpened'] = sharpened.copy()
    
    # Apply edge detection
    edges = edge_detection(sharpened)
    if debug:   
        debug_images['4_edges'] = edges.copy()
    
    # Apply morphological operations
    morph_image = morphological_operations(edges)
    if debug:
        debug_images['5_morphology'] = morph_image.copy()
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        morph_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 51, 9
    )
    if debug:
        debug_images['6_threshold'] = thresh.copy()
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    if debug:
        debug_images['7_denoised'] = denoised.copy()
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    if debug:
        debug_images['8_enhanced'] = enhanced.copy()

    # Apply histogram matching if reference image is provided
    if reference_image is not None:
        enhanced = histogram_matching(enhanced, reference_image)
        if debug:
            debug_images['9_histogram_matched'] = enhanced.copy()
    
    return enhanced, debug_images if debug else None

# Barcode detection and decoding function
def detect_and_decode_barcode(image, reference_image=None, debug=False):
    original_image = image.copy()
    decoded_objects = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    processing_methods = [
        lambda img: img,  # Original
        lambda img: cv2.GaussianBlur(img, (5, 5), 0),  # Blur
        lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # Otsu
        lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),  # Adaptive
    ]
    
    enhanced = None
    for process in processing_methods:
        processed = process(gray)
        
        for scale in [1.0, 1.5, 2.0]:
            if scale != 1.0:
                width = int(processed.shape[1] * scale)
                height = int(processed.shape[0] * scale)
                dim = (width, height)
                scaled = cv2.resize(processed, dim, interpolation=cv2.INTER_CUBIC)
            else:
                scaled = processed
            
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
    
    if not decoded_objects:
        enhanced, debug_images = enhance_barcode_image(image, reference_image, debug=debug)
        if debug and debug_images:
            for step_name, img in debug_images.items():
                cv2.imwrite(f"{step_name}.png", img)
        
        decoded_objects = decode(enhanced)
    
    if decoded_objects:
        for obj in decoded_objects:
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
            
            barcode_data = obj.data.decode('utf-8')
            barcode_type = obj.type
            x, y = points[0]
            cv2.putText(original_image, f"{barcode_data} ({barcode_type})",
                      (int(x), int(y) - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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

