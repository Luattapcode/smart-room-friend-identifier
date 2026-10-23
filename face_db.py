import os
import cv2

def load_friend_database(detector, recognizer, folder="friends"):
    database = {} # Dictionary lưu: { "tên": feature_vector }

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"Thư mục '{folder}' chưa có ảnh nào. Hãy bỏ ảnh bạn bè vào đây!")
        return database

    for filename in os.listdir(folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Lấy tên bạn bè từ tên file (ví dụ "nam.jpg" -> "nam")
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(folder, filename)
        
        img = cv2.imread(image_path)
        if img is None:
            continue

        # Đặt kích thước ảnh để detector quét
        h, w, _ = img.shape
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)

        if faces is not None:
            # 1. Căn chỉnh khuôn mặt
            aligned_face = recognizer.alignCrop(img, faces[0])
            # 2. Rút trích vector 128 số thực đặc trưng
            feature = recognizer.feature(aligned_face)
            
            database[name] = feature
            print(f"-> Đã nạp dữ liệu khuôn mặt: {name}")
        else:
            print(f"Cảnh báo: Không tìm thấy mặt trong ảnh {filename}")

    return database