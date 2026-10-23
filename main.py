import cv2
from detector import get_face_detector, get_face_recognizer
from face_db import load_friend_database

# 1. KHỞI TẠO CÁC MÔ HÌNH AI
print("Đang khởi tạo các mô hình AI...")
detector = get_face_detector()
recognizer = get_face_recognizer()

# 2. NẠP DỮ LIỆU BẠN BÈ TỪ THƯ MỤC 'friends'
print("Đang nạp dữ liệu bạn bè...")
friends_db = load_friend_database(detector, recognizer, folder="friends")
print(f"-> Đã sẵn sàng nhận diện {len(friends_db)} người bạn: {list(friends_db.keys())}\n")

# Ngưỡng nhận diện (Cosine threshold chuẩn của SFace là 0.363)
COSINE_THRESHOLD = 0.363

# 3. MỞ WEBCAM
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Lỗi: Không mở được camera!")
    exit()

print("Hệ thống nhận diện phòng bắt đầu hoạt động! Bấm 'q' để dừng.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    detector.setInputSize((w, h))

    # Phát hiện khuôn mặt
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            # 1. Trích xuất đặc trưng của khuôn mặt đang trước camera
            aligned_face = recognizer.alignCrop(frame, face)
            current_feature = recognizer.feature(aligned_face)

            # 2. So sánh với danh sách bạn bè trong cơ sở dữ liệu
            best_match_name = "Unknown"
            highest_score = 0.0

            for friend_name, friend_feature in friends_db.items():
                # Tính độ khớp Cosine (càng gần 1.0 càng giống nhau)
                score = recognizer.match(current_feature, friend_feature, cv2.FaceRecognizerSF_FR_COSINE)
                if score > highest_score:
                    highest_score = score
                    if score >= COSINE_THRESHOLD:
                        best_match_name = friend_name

            # 3. Toạ độ để vẽ
            x, y, box_w, box_h = map(int, face[:4])

            # 4. Hiển thị kết quả: Màu xanh nếu là bạn bè, Màu đỏ nếu là người lạ
            if best_match_name != "Unknown":
                color = (0, 255, 0) # Xanh lá (BGR)
                label = f"{best_match_name} ({highest_score:.2f})"
            else:
                color = (0, 0, 255) # Đỏ (BGR)
                label = f"Unknown ({highest_score:.2f})"

            # Vẽ khung chữ nhật
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 2)
            
            # Vẽ nền chữ và in tên lên góc trên khung mặt
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Room Face Identifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()