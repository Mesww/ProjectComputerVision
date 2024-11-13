import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image
import matplotlib.pyplot as plt

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """ปรับความสว่างและคอนทราสต์ของภาพ"""
    new_image = cv2.convertScaleAbs(image, alpha=1 + contrast/100, beta=brightness)
    return new_image

def histogram_matching(source, reference):
    """ทำ Histogram Matching"""
    matched = cv2.equalizeHist(source)
    return matched

def fourier_filter(image, filter_type='low', cutoff=50):
    """Fourier Transform สำหรับ Low-pass, High-pass, และ Band-pass Filtering"""
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols, 2), np.uint8)
    if filter_type == 'low':
        cv2.circle(mask, (ccol, crow), cutoff, (0, 0), -1)
    elif filter_type == 'high':
        mask = 1 - mask
        cv2.circle(mask, (ccol, crow), cutoff, (0, 0), -1)

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)

def apply_edge_detection(image):
    """ตรวจจับขอบภาพด้วย Sobel, Prewitt, และ Canny"""
    sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    prewitt = cv2.filter2D(image, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
    canny = cv2.Canny(image, 100, 200)
    return sobel, prewitt, canny

def otsu_threshold(image):
    """แบ่งส่วนภาพด้วย Otsu’s Thresholding"""
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu

def color_segmentation(image):
    """แบ่งส่วนภาพด้วย HSV Color Space"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 50, 50])
    upper = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented

def geometric_transformations(image):
    """การแปลงภาพเชิงเรขาคณิต: Translation, Scaling, Rotation"""
    height, width = image.shape[:2]

    # Translation
    M_translate = np.float32([[1, 0, 100], [0, 1, 50]])
    translated = cv2.warpAffine(image, M_translate, (width, height))

    # Scaling
    scaled = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # Rotation
    M_rotate = cv2.getRotationMatrix2D((width // 2, height // 2), 45, 1)
    rotated = cv2.warpAffine(image, M_rotate, (width, height))

    return translated, scaled, rotated

def interpolation_methods(image):
    """การทำ Interpolation แบบ Nearest, Bilinear, Bicubic"""
    nearest = cv2.resize(image, (300, 300), interpolation=cv2.INTER_NEAREST)
    bilinear = cv2.resize(image, (300, 300), interpolation=cv2.INTER_LINEAR)
    bicubic = cv2.resize(image, (300, 300), interpolation=cv2.INTER_CUBIC)
    return nearest, bilinear, bicubic

def image_registration(image1, image2):
    """การทำ Image Registration ด้วย Feature Matching"""
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None)
    return result

# ตัวอย่างการเรียกใช้
if __name__ == "__main__":
    image = cv2.imread("./public/assets/Code-39.jpg", cv2.IMREAD_GRAYSCALE)

    # ปรับความสว่างและคอนทราสต์
    enhanced_image = adjust_brightness_contrast(image, brightness=30, contrast=30)

    # ตรวจจับขอบ
    sobel, prewitt, canny = apply_edge_detection(enhanced_image)

    # การทำ Fourier Filtering
    filtered_image = fourier_filter(enhanced_image, filter_type='low')

    # การแบ่งส่วนภาพด้วย Otsu
    segmented_image = otsu_threshold(enhanced_image)

    # การทำ Image Registration
    registered_image = image_registration(image, enhanced_image)

    # แสดงผล
    cv2.imshow("Enhanced", enhanced_image)
    cv2.imshow("Canny", canny)
    cv2.imshow("Fourier Filtered", filtered_image)
    cv2.imshow("Otsu Threshold", segmented_image)
    cv2.imshow("Registered Image", registered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()