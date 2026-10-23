# File: detector.py
import os
import urllib.request
import cv2

def get_face_detector():
    model_dir = "models"
    model_path = os.path.join(model_dir, "face_detection_yunet.onnx")
    model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print("Đang tải mô hình YuNet...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Tải xong!")

    return cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(320, 320),
        score_threshold=0.6
    )
def get_face_recognizer():
    model_dir = "models"
    model_path = os.path.join(model_dir, "face_recognition_sface.onnx")
    model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    if not os.path.exists(model_path):
        print("Đang tải mô hình SFace (khoảng 38 MB)... Vui lòng đợi trong giây lát.")
        urllib.request.urlretrieve(model_url, model_path)
        print("Tải SFace thành công!")
    # Khởi tạo FaceRecognizerSF của OpenCV
    return cv2.FaceRecognizerSF.create(
        model=model_path,
        config=""
    )