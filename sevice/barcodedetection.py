from typing import Tuple
import cv2
import numpy as np
from pyzbar.pyzbar import decode

def enhance_barcode_image(image: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Gentler preprocessing pipeline for barcode images that preserves more information.
    
    Args:
        image: Input image (BGR or grayscale)
        debug: If True, returns intermediate processing steps
    
    Returns:
        Tuple of (enhanced image, debug_images dict if debug=True)
    """
    debug_images = {}
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if debug:
        debug_images['1_grayscale'] = gray.copy()

    # 1. Gentle denoising with smaller parameters
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    if debug:
        debug_images['2_denoised'] = denoised.copy()

    # 2. Moderate contrast enhancement using CLAHE with smaller clip limit
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    enhanced_contrast = clahe.apply(denoised)
    if debug:
        debug_images['3_contrast_enhanced'] = enhanced_contrast.copy()

    # 3. Gentle sharpening
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]]) / 9
    sharpened = cv2.filter2D(enhanced_contrast, -1, kernel)
    if debug:
        debug_images['4_sharpened'] = sharpened.copy()

    # 4. Adaptive thresholding with more moderate parameters
    thresh = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,  # Larger block size
        5    # Smaller C constant
    )
    if debug:
        debug_images['5_threshold'] = thresh.copy()

    # 5. Minimal morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    if debug:
        debug_images['6_final'] = cleaned.copy()

    return (cleaned, debug_images) if debug else (cleaned, None)

def detect_and_decode_barcode(image, debug=False):
    """
    Detect and decode barcodes using both original and enhanced images with multiple attempts
    
    Args:
        image: Input image (BGR format)
        debug: If True, returns debug information
    
    Returns:
        Tuple of (enhanced image, decoded objects)
    """
    # First try with original image
    enhanced, debug_images = enhance_barcode_image(image, debug=debug)
    if debug_images:
        for step_name, img in debug_images.items():
            cv2.imwrite(f"{step_name}.png", img)
    cv2.imwrite("public/assets/processed_barcode.png", enhanced)
    decoded_objects = decode(image)
    
    if not decoded_objects:
        # Get enhanced image
        
        # Try different processing variations for better detection
        attempts = [
            enhanced,  # Try enhanced image as is
            cv2.bitwise_not(enhanced),  # Try inverted image
            cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # Try Otsu thresholding
        ]
        
        for attempt in attempts:
            decoded_objects = decode(attempt)
            if decoded_objects:
                break
    
    if decoded_objects:
        # Draw detected barcodes
        for obj in decoded_objects:
            barcode_data = obj.data.decode('utf-8')
            barcode_type = obj.type
            points = obj.polygon

        if points is not None and len(points) > 4:
            # Use convex hull if there are more than four points
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            points = [tuple(point[0]) for point in hull]

                # Add text annotation
                x, y = points[0]
                cv2.putText(image, f"{barcode_data} ({barcode_type})", 
                          (int(x), int(y) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"Found {barcode_type} barcode: {barcode_data}")
    else:
        print("No barcodes detected in the image")

    return enhanced if 'enhanced' in locals() else image, decoded_objects
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
