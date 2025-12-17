import numpy as np
import cv2
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import time
import os

thd = 0.75  # Threshold
font = cv2.FONT_HERSHEY_SIMPLEX

# Load the current model
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'model.h5')
mdl = tf.keras.models.load_model(model_path)

def preprocess_img(imgBGR, erode_dilate=True):
    rows, cols, _ = imgBGR.shape
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    Bmin = np.array([100, 43, 46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)

    Rmin1 = np.array([0, 43, 46])
    Rmax1 = np.array([10, 255, 255])
    img_Rbin1 = cv2.inRange(imgHSV, Rmin1, Rmax1)

    Rmin2 = np.array([156, 43, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin2 = cv2.inRange(imgHSV, Rmin2, Rmax2)
    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    img_bin = np.maximum(img_Bbin, img_Rbin)

    if erode_dilate:
        kernelErosion = np.ones((3, 3), np.uint8)
        kernelDilation = np.ones((3, 3), np.uint8)
        img_bin = cv2.erode(img_bin, kernelErosion, iterations=1)
        img_bin = cv2.dilate(img_bin, kernelDilation, iterations=1)

    return img_bin

def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def getClassName(classNo):
    classNames = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return classNames[classNo] if classNo < len(classNames) else 'Unknown'

def predict_image(img):
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1).astype('float32')
    predictions = mdl.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.amax(predictions)
    return classIndex, probabilityValue

if __name__ == '__main__':
    input_video = input("Enter image or video path or press enter for webcam:")
    if input_video == '':
        cap = cv2.VideoCapture(0)
        is_image = False
    elif input_video.endswith(('.jpg', '.png', '.jpeg', '.webp')):
        cap = cv2.imread(input_video)
        is_image = True
    else:
        cap = cv2.VideoCapture(input_video)
        is_image = False

    cols = cap.shape[1] if is_image else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = cap.shape[0] if is_image else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def process_frame(frame):
        img_bin = preprocess_img(frame, False)
        min_area = img_bin.shape[0] * img_bin.shape[1] / (25 * 25)
        rects = contour_detect(img_bin, min_area=min_area)
        img_bbx = frame.copy()
        for rect in rects:
            xc = int(rect[0] + rect[2] / 2)
            yc = int(rect[1] + rect[3] / 2)
            size = max(rect[2], rect[3])
            x1 = max(0, int(xc - size / 2))
            y1 = max(0, int(yc - size / 2))
            x2 = min(cols, int(xc + size / 2))
            y2 = min(rows, int(yc + size / 2))
            if rect[2] > 100 and rect[3] > 100:
                cv2.rectangle(img_bbx, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
                crop_img = frame[y1:y2, x1:x2]
                classIndex, probabilityValue = predict_image(crop_img)
                if probabilityValue > thd:
                    cv2.putText(img_bbx, str(getClassName(classIndex)), 
                                (rect[0], rect[1] - 10), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(img_bbx, "Accuracy: " + str(round(probabilityValue * 100, 2)) + "%", 
                                (rect[0], rect[1] - 40), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        return img_bbx

    if is_image:
        processed_frame = process_frame(cap)
        if processed_frame.shape[0] > 1920 and processed_frame.shape[1] > 1080:
            resized_frame = cv2.resize(processed_frame, (1920, 1080))
        else:
            resized_frame = processed_frame
        cv2.imshow("detect result", resized_frame)
        while True:
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    else:
        with ThreadPoolExecutor(max_workers=4) as executor:
            prev_time = time.time()
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                future = executor.submit(process_frame, frame)
                processed_frame = future.result()
                if processed_frame.shape[0] > 1920 and processed_frame.shape[1] > 1080:
                    resized_frame = cv2.resize(processed_frame, (1920, 1080))
                else:
                    resized_frame = processed_frame
                cv2.imshow("detect result", resized_frame)

                frame_count += 1
                curr_time = time.time()
                elapsed_time = curr_time - prev_time
                if elapsed_time >= 1.0:
                    fps = frame_count / elapsed_time
                    print(f"FPS: {fps}")
                    prev_time = curr_time
                    frame_count = 0

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    if not is_image:
        cap.release()
    cv2.destroyAllWindows()
