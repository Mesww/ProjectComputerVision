import cv2
import numpy as np
from pyzbar.pyzbar import decode
import os
import sys
from PIL import Image

def enhance_barcode_image(image):
    """
    Enhance image quality for better barcode detection
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Basic image enhancements
    # 1. Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    
    # 2. Noise reduction
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # 3. Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # C constant
    )
    
    # 4. Morphological operations
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

def detect_and_decode_barcode(image):
    decoded_objects = decode(image)

    if not decoded_objects:
        # Try enhanced image if no barcodes are detected
        enhanced = enhance_barcode_image(image)
        decoded_objects = decode(enhanced)
    
    for obj in decoded_objects:
        barcode_data = obj.data.decode('utf-8')
        barcode_type = obj.type
        points = obj.polygon

        points = obj.polygon

        if points is not None and len(points) > 4:
            # Use convex hull if there are more than four points
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            points = [tuple(point[0]) for point in hull]

        if points is not None:
            # Draw the bounding box around the barcode
            n = len(points)
            for j in range(n):
                pt1 = (int(points[j][0]), int(points[j][1]))
                pt2 = (int(points[(j+1) % n][0]), int(points[(j+1) % n][1]))
                cv2.line(image, pt1, pt2, (0, 255, 0), 3)

            # Put barcode data and type on the image
            x, y = points[0]
            cv2.putText(image, f"{barcode_data} ({barcode_type})", (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Found {barcode_type} barcode: {barcode_data}")

    return image, decoded_objects

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


# def detect_barcode(image):
#     # Step 1: Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Step 2: Compute the gradient of the image
#     gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
#     gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    
#     # Subtract gradients
#     gradient = cv2.subtract(gradX, gradY)
#     gradient = cv2.convertScaleAbs(gradient)

#     # Step 3: Blur and threshold the image
#     blurred = cv2.blur(gradient, (9, 9))
#     _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

#     # Step 4: Erode and dilate the image to clean up
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
#     closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#     # Perform series of erosions and dilations
#     closed = cv2.erode(closed, None, iterations=4)
#     closed = cv2.dilate(closed, None, iterations=4)

#     # Step 5: Find contours in the image
#     contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if contours:
#         # Sort contours by area, largest first
#         c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        
#         # Compute the rotated bounding box of the largest contour
#         rect = cv2.minAreaRect(c)
#         box = cv2.boxPoints(rect)
#         box = np.int32(box)

#         # Draw the bounding box on the image
#         cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

#     return image