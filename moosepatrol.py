#!/usr/bin/env python3  # Добавьте эту строку для запуска скрипта как исполняемого

import cv2
from ultralytics import YOLO
import numpy as np
import time
import os
import rospy
from clover.srv import Navigate, NavigateGlobal, GetTelemetry, SetAttitude
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Trigger

# -------------------- НАСТРОЙКИ --------------------
MODEL_PATH = "yolov8s.pt"  # Путь к вашей модели YOLOv8 (скачайте или обучите)
CONFIDENCE_THRESHOLD = 0.6  # Порог уверенности для фильтрации обнаружений
CLASS_ID_OF_INTEREST = 67 # ID класса "deer" (олень).  Зависит от датасета на котором обучалась модель.  Проверьте файл data.yaml вашей модели.
# ---------------------------------------------------

# Переменные для работы с ROS
rospy.init_node('deer_detection_and_flight', anonymous=True)
bridge = CvBridge()
navigate_srv = rospy.ServiceProxy('navigate', Navigate)
navigate_global_srv = rospy.ServiceProxy('navigate_global', NavigateGlobal)
get_telemetry = rospy.ServiceProxy('get_telemetry', GetTelemetry)
set_attitude = rospy.ServiceProxy('set_attitude', SetAttitude)
land_srv = rospy.ServiceProxy('land', Trigger)
# Переменные для работы с дроном

def load_model(model_path):
    """Загружает модель YOLOv8."""
    model = YOLO(model_path)
    return model

def image_callback(data):
    """Обработчик сообщений с изображениями от камеры."""
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")  # Преобразование в OpenCV формат
    except CvBridgeError as e:
        print(e)
        return

    #  Основная логика обработки изображения и обнаружения
    start_time = time.time()
    results = detect_objects(model, cv_image)
    detections = get_deer_detections(results)
    end_time = time.time()

    if detections:
        timestamp = int(time.time())
        save_screenshot(cv_image, detections, timestamp)  # Сохраняем скриншот

        # ---------  КОД ДЛЯ УПРАВЛЕНИЯ ПОЛЕТОМ (после обнаружения) --------
        print("Олень обнаружен! Выполняем действия...")
        try:
            # 1. Остановиться и зависнуть (если дрон летит)
            # (Clover обычно автоматически зависает при получении команд)
            # time.sleep(2)

            # 2. Рассчитать координаты для полета к оленю
            # (Здесь нужно добавить код для расчета координат, используя
            #  информацию о положении оленя на изображении, высоте полета дрона,
            #  FOV камеры и т.д.)
            #  Пример:
            #  x_offset, y_offset = calculate_offset_from_detection(detections[0]['box'], cv_image.shape)
            #  latitude, longitude, altitude = calculate_target_coordinates(x_offset, y_offset, текущие_координаты_дрона, высота_полета)

            # 3. Получение текущих координат дрона
            telemetry = get_telemetry()
            current_latitude = telemetry.latitude
            current_longitude = telemetry.longitude
            current_altitude = telemetry.altitude

            # 4.  Перелет к точке интереса (Пример, движение по прямой в метрах относительно текущего положения)
            # x_offset = 2 #Сдвиг по X (метр)
            # y_offset = 2 #Сдвиг по Y (метр)
            # z_offset = 1 #Сдвиг по Z (метр)
            #
            # navigate_srv(x=x_offset, y=y_offset, z=z_offset, speed=0.5) #  движение по X,Y,Z
            # rospy.sleep(5) # Ждем 5 секунд

            # 5.  Посадка (остановка дрона)
            print("Посадка...")
            land_srv()

        except Exception as e:
            print(f"Ошибка при управлении полетом: {e}")
        # ---------------------------------------------------

    annotated_frame = draw_detections(cv_image.copy(), detections)  # Отображение на экране (для отладки)
    cv2.imshow("Определение оленей", annotated_frame)  # Отображение в окне
    cv2.waitKey(1)  # Необходимо для отображения окна OpenCV

    print(f"Время обработки: {end_time - start_time:.2f} сек")

def detect_objects(model, frame):
    """Выполняет обнаружение объектов на кадре."""
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)  # verbose=False убирает вывод в консоль
    return results

def get_deer_detections(results, class_id=CLASS_ID_OF_INTEREST):
    """Фильтрует обнаружения, оставляя только оленей."""
    detections = []
    if results and results[0].boxes is not None:  # Проверка, что обнаружения есть
        for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            if cls == class_id:
                detections.append({
                    'box': box.cpu().numpy().astype(int), # преобразуем в массив numpy и int
                    'confidence': conf.cpu().numpy(),  #  преобразуем в число
                })
    return detections

def draw_detections(frame, detections):
    """Рисует рамки и метки на кадре."""
    for detection in detections:
        box = detection['box']
        confidence = detection['confidence']
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зеленый цвет
        cv2.putText(frame, f"Олень {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def save_screenshot(frame, detections, timestamp):
    """Сохраняет скриншот с обнаружениями."""
    filename = f"deer_screenshot_{timestamp}.jpg"
    annotated_frame = draw_detections(frame.copy(), detections)  #  Создаем копию кадра, чтобы не портить исходный
    cv2.imwrite(filename, annotated_frame)
    print(f"Скриншот сохранен: {filename}")


def main():
    """Основной цикл работы."""
    global model # Делаем model глобальной переменной
    model = load_model(MODEL_PATH)

    # Подписываемся на топик с изображениями
    rospy.Subscriber("main_camera/image_raw", Image, image_callback) #  camera/image_raw  или  camera/image
    # camera_info -  для калибровки камеры
    rospy.spin()  # Ждем сообщений от камеры

    print("Завершено.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
