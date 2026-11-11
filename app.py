import cv2
import threading
import time
import gc
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# ป้องกันเบราว์เซอร์จำ Cache
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# โหลดโมเดล YOLOv8
try:
    model = YOLO('best.pt')
    print("Loaded YOLOv8 model (best.pt) successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# พจนานุกรมจับคู่คำแปลภาษาไทย (ครอบคลุมทั้งชื่อโฟลเดอร์และชื่อคลาส)
TRANSLATION_DICT = {
    # หมวดคำศัพท์ตามโฟลเดอร์ Dataset
    "A sharp, shooting headache": "ปวดหัวแปล๊บๆ เหมือนไฟช็อต",
    "Can I have some medicine": "ขอยาหน่อยได้ไหม",
    "Experiencing pain": "มีอาการปวด/เจ็บ",
    "faint": "เป็นลม / หน้ามืด",
    "Gastritis": "โรคกระเพาะอาหารอักเสบ",
    "have a cold": "เป็นหวัด",
    "headache": "ปวดหัว",
    "hospital": "โรงพยาบาล",
    "Menstruation": "ปวดประจำเดือน",
    "radiating headache": "ปวดหัวร้าวไปจุดอื่น",
    "Throbbing headache": "ปวดหัวตุ๊บๆ",
    "toothache": "ปวดฟัน",
    "Unconscious": "หมดสติ / ไม่รู้สึกตัว",
    
    # หมวดคำศัพท์ทั่วไปเพิ่มเติม
    "sore_throat": "เจ็บคอ",
    "stomachache": "ปวดท้อง",
    "dizzy": "เวียนหัว",
    "cough": "ไอ",
    "nausea": "คลื่นไส้ อยากอาเจียน",
    "not_feeling_well": "รู้สึกไม่ค่อยสบาย"
}

current_sentence = "กำลังรอสัญญาณมือ..."
hand_detected = False
output_frame = None
lock = threading.Lock()

def translate_to_thai(text):
    """ฟังก์ชั่นค้นหาคำแปลภาษาไทยแบบยืดหยุ่น"""
    if not text:
        return "กำลังรอสัญญาณมือ..."
    
    # 1. ค้นหาแบบตรงตัวก่อน
    if text in TRANSLATION_DICT:
        return TRANSLATION_DICT[text]
    
    # 2. ค้นหาแบบไม่สนตัวพิมพ์เล็ก-ใหญ่ หรืออักขระพิเศษ
    clean_text = str(text).strip().lower().replace("_", " ")
    for k, v in TRANSLATION_DICT.items():
        if k.lower().replace("_", " ") == clean_text:
            return v
            
    # 3. ถ้าไม่พบในพจนานุกรมจริงๆ ให้แสดงข้อความเดิม
    return text

def camera_loop():
    global current_sentence, hand_detected, output_frame
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    no_detect_counter = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue

        frame_count += 1
        frame = cv2.flip(frame, 1)

        if model is not None:
            results = model.predict(source=frame, conf=0.35, verbose=False)
            annotated_frame = results[0].plot()

            if len(results[0].boxes) > 0:
                hand_detected = True
                no_detect_counter = 0
                
                # ดึงคลาสที่ทำนายได้
                box = results[0].boxes[0]
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                # แปลงเป็นภาษาไทยก่อนส่งออกหน้าจอ
                current_sentence = translate_to_thai(class_name)
            else:
                no_detect_counter += 1
                if no_detect_counter > 10:
                    hand_detected = False
                    current_sentence = "กำลังรอสัญญาณมือ..."
        else:
            annotated_frame = frame

        if frame_count % 100 == 0:
            gc.collect()

        with lock:
            output_frame = annotated_frame.copy()

        time.sleep(0.02)

    cap.release()

t = threading.Thread(target=camera_loop)
t.daemon = True
t.start()

def generate_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sentence')
def get_sentence():
    return jsonify({
        "full_sentence": current_sentence,
        "hand_detected": hand_detected
    })

@app.route('/get_dictionary')
def get_dictionary():
    vocab_list = [{"eng": k, "th": v} for k, v in TRANSLATION_DICT.items()]
    return jsonify(vocab_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)