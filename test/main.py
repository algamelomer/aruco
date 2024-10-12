import cv2 as cv
from cv2 import aruco
import numpy as np
import time

# تحميل قاموس الماركرات
marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

# إعدادات كشف الماركرات المتقدمة
param_markers = aruco.DetectorParameters_create()
param_markers.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX  # تحسين زوايا الماركر
param_markers.minDistanceToBorder = 5  # تحسين دقة الماركرات القريبة من الحواف

# فتح كاميرا Iriun Webcam (عادةً تكون ID=1)
cup = cv.VideoCapture(1)  # استخدم ID=1 لكاميرا Iriun

# تحقق من فتح الكاميرا بنجاح
if not cup.isOpened():
    print("خطأ: لا يمكن فتح الكاميرا")
    exit()

# تعيين دقة الكاميرا المطلوبة
frame_width = int(cup.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cup.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_center = np.array([frame_width // 2, frame_height // 2])  # نقطة مركز الإطار

# تعيين عتبة المسافة لاعتبار الكاميرا متعامدة
threshold = 50  # يمكنك تغيير العتبة حسب الحاجة

# الحصول على الوقت الأولي لحساب FPS
prev_time = time.time()
fps_list = []

while True:
    ret, frame = cup.read()
    if not ret:
        print("خطأ: لم يتم التقاط إطار")
        break
    
    # تحويل الإطار إلى الصورة الرمادية
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # اكتشاف الماركرات في الإطار
    marker_corners, marker_IDs, _ = aruco.detectMarkers(
        gray_frame,
        marker_dict,
        parameters=param_markers
    )
    
    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            # تحويل الزوايا إلى إحداثيات صحيحة
            corners = corners.reshape(4, 2).astype(int)

            # حساب نقطة المركز للماركر
            center = np.mean(corners, axis=0).astype(int)
            
            # حساب المسافة من مركز الإطار
            distance = np.linalg.norm(center - frame_center)
            
            # تغيير لون الإطار بناءً على المسافة من المركز
            color = (0, 255, 0) if distance < threshold else (0, 255, 255)  # أخضر أو أصفر بناءً على المسافة
            
            # رسم السهم للإشارة إلى اتجاه الحركة إذا كانت الكاميرا غير متعامدة
            if distance >= threshold:
                direction_vector = frame_center - center
                arrow_tip = center + direction_vector // 2  # رأس السهم
                cv.arrowedLine(frame, tuple(center), tuple(arrow_tip), (255, 0, 0), 3, tipLength=0.3)
            
            # رسم الخطوط حول الماركرات
            cv.polylines(frame, [corners], True, color, 4, cv.LINE_AA)
            
            # وضع النص الذي يحتوي على ID الماركر مع خلفية
            top_right = corners[0].ravel()
            cv.rectangle(frame, (top_right[0], top_right[1] - 25), (top_right[0] + 100, top_right[1] + 5), (0, 0, 0), -1)
            cv.putText(frame,
                       f"ID: {ids[0]}",
                       tuple(top_right),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1.0,
                       (255, 255, 255),
                       2,
                       cv.LINE_AA
                       )
            
            # رسم دائرة في نقطة المركز
            cv.circle(frame, tuple(center), 5, (0, 0, 255), -1)  # دائرة حمراء

            # عرض إحداثيات مركز الماركر والمسافة من مركز الإطار
            cv.putText(frame, f"Marker Center: {center}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(frame, f"Distance from Center: {int(distance)} px", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # حساب كمية الحركة المطلوبة لجعل الكاميرا مقابلة لمركز الماركر
            movement_needed = frame_center - center
            cv.putText(frame, f"Move X: {movement_needed[0]} px, Y: {movement_needed[1]} px", (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    # رسم دائرة تمثل مركز الإطار
    cv.circle(frame, tuple(frame_center), 5, (255, 0, 0), -1)  # دائرة زرقاء في مركز الإطار
    
    # حساب FPS باستخدام قائمة لتخزين الفروقات الزمنية
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    fps_list.append(fps)
    if len(fps_list) > 30:  # تخزين آخر 30 قيمة لحساب المتوسط
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list)

    # عرض معدل الإطارات على الشاشة
    cv.putText(frame, f"FPS: {int(avg_fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    # عرض الإطار
    cv.imshow("frame", frame)
    
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# إغلاق الكاميرا وإغلاق النوافذ
cup.release()
cv.destroyAllWindows()
