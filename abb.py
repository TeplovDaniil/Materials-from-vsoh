import streamlit as st
import cv2
import numpy as np
import os
import json
from PIL import Image
import face_recognition
import time


# Конфигурация
BASE_DIR = "face_database"
os.makedirs(BASE_DIR, exist_ok=True)
METADATA_PATH = os.path.join(BASE_DIR, 'metadata.json')

# Функции работы с данными
def load_metadata():
    try:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            for entry in metadata:
                entry['encoding'] = np.array(entry['encoding'])
            return metadata
        return []
    except Exception as e:
        st.error(f"Ошибка загрузки метаданных: {str(e)}")
        return []

def save_metadata(metadata):
    try:
        temp_metadata = []
        for entry in metadata:
            temp_entry = entry.copy()
            temp_entry['encoding'] = temp_entry['encoding'].tolist()
            temp_metadata.append(temp_entry)
        
        with open(METADATA_PATH, 'w') as f:
            json.dump(temp_metadata, f, indent=2)
    except Exception as e:
        st.error(f"Ошибка сохранения метаданных: {str(e)}")

def process_frame(frame):
    try:
        # Уменьшаем размер кадра для обработки
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Обнаружение лиц с обработкой исключений
        face_locations = []
        try:
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        except Exception as e:
            st.error(f"Ошибка обнаружения лиц: {str(e)}")
            return frame, []

        # Масштабирование координат с проверкой границ
        scaled_locations = []
        for (top, right, bottom, left) in face_locations:
            try:
                top = max(0, min(top*2, frame.shape[0]-1))
                right = max(0, min(right*2, frame.shape[1]-1))
                bottom = max(0, min(bottom*2, frame.shape[0]-1))
                left = max(0, min(left*2, frame.shape[1]-1))
                
                if (bottom - top) < 20 or (right - left) < 20:
                    continue  # Пропускаем слишком маленькие области
                    
                scaled_locations.append((top, right, bottom, left))
            except Exception as e:
                st.error(f"Ошибка масштабирования координат: {str(e)}")

        # Отрисовка рамок
        for (top, right, bottom, left) in scaled_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        return frame, scaled_locations
    
    except Exception as e:
        st.error(f"Ошибка обработки кадра: {str(e)}")
        return frame, []

def run_camera():
    st.write("### Режим реального отслеживания")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    FRAME_WINDOW = st.image([])
    
    # Управление состоянием
    if 'capture' not in st.session_state:
        st.session_state.capture = False
    if 'stop' not in st.session_state:
        st.session_state.stop = False
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Остановить камеру"):
            st.session_state.stop = True
    with col2:
        if st.button("Сделать снимок"):
            st.session_state.capture = True

    metadata = load_metadata()
    known_encodings = [entry['encoding'] for entry in metadata]

    while cap.isOpened() and not st.session_state.stop:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            st.error("Ошибка получения кадра")
            break

        # Обработка кадра
        processed_frame, face_locations = process_frame(frame.copy())
        
        # Конвертация в RGB для Streamlit
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        # Обработка захвата фото
        if st.session_state.capture:
            try:
                if not face_locations:
                    raise ValueError("Лица не обнаружены в момент съемки")
                
                # Используем первое обнаруженное лицо
                top, right, bottom, left = face_locations[0]
                
                # Проверка координат
                if any([coord < 0 for coord in [top, right, bottom, left]]):
                    raise ValueError("Некорректные координаты лица")
                
                # Вырезаем область с лицом
                face_img = frame_rgb[top:bottom, left:right]
                if face_img.size == 0:
                    raise ValueError("Пустая область лица")
                
                # Конвертация в RGB для face_recognition
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # Получение кодировки
                face_encodings = face_recognition.face_encodings(
                    face_img_rgb,
                    known_face_locations=[(0, face_img.shape[1], face_img.shape[0], 0)],
                    num_jitters=1,
                    model="small"
                )
                
                if not face_encodings:
                    raise ValueError("Не удалось извлечь признаки лица")
                
                face_encoding = face_encodings[0]

                # Сравнение с базой
                matches = face_recognition.compare_faces(
                    known_encodings, 
                    face_encoding,
                    tolerance=0.5
                )

                if True in matches:
                    match_index = matches.index(True)
                    metadata[match_index]['count'] += 1
                    st.success(f"Посещений: {metadata[match_index]['count']}")
                else:
                    filename = f"user_{len(metadata)}.jpg"
                    save_path = os.path.join(BASE_DIR, filename)
                    Image.fromarray(face_img).save(save_path)
                    
                    metadata.append({
                        'filename': filename,
                        'encoding': face_encoding,
                        'count': 1
                    })
                    st.success("Новый пользователь добавлен!")
                
                save_metadata(metadata)
                known_encodings = [entry['encoding'] for entry in metadata]

            except Exception as e:
                st.error(f"Ошибка обработки: {str(e)}")
            
            finally:
                st.session_state.capture = False

        # Ограничение FPS
        elapsed_time = time.time() - start_time
        time.sleep(max(0.033 - elapsed_time, 0))

    cap.release()
    cv2.destroyAllWindows()
#
def Evkl_distance(model_predictions, folder_path):
    results = {}

    # 1. Получаем список файлов в папке
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    if not files:
        print(f"Ошибка: Папка '{folder_path}' пуста.")
        return None

    # 2. Итерируемся по файлам и сравниваем с предсказаниями
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # ***Здесь должна быть логика загрузки данных из файла***
        # В этом примере мы создаем случайные данные для имитации
        # Предположим, что данные в файле имеют ту же размерность, что и предсказания
        data_from_file = np.random.rand(*model_predictions.shape)  # Замените на реальную загрузку данных

        # 3. Вычисляем евклидово расстояние
        distance = euclidean_distances(model_predictions.reshape(1, -1), data_from_file.reshape(1, -1))[0][0]

        results[file_name] = distance

    return results


def show_database():
    st.write("### База пользователей")
    metadata = load_metadata()
    
    if not metadata:
        st.warning("База данных пуста")
        return
    
    cols = st.columns(3)
    for idx, entry in enumerate(metadata):
        with cols[idx % 3]:
            img_path = os.path.join(BASE_DIR, entry['filename'])
            try:
                st.image(img_path, 
                        caption=f"Посещений: {entry['count']}",
                        use_column_width=True)
                st.write(f"ID: {entry['filename'].split('.')[0]}")
            except Exception as e:
                st.error(f"Ошибка загрузки изображения {entry['filename']}: {str(e)}")
                

def main():
    st.title("Система распознавания лиц v2.0")
    
    menu = st.selectbox("Меню", 
                       ["Инструкция", "Камера", "База данных"])
    
    if menu == "Инструкция":
        st.write("""
        ### Руководство пользователя:
        1. Перейдите в раздел **Камера**
        2. Убедитесь, что лицо находится в зеленой рамке
        3. Нажмите **Сделать снимок** для фиксации
        4. Просматривайте статистику в разделе **База данных**

        **Советы:**
        - Хорошее освещение лица
        - Позиционирование по центру кадра
        - Отсутствие масок и очков
        """)
        
    elif menu == "Камера":
        run_camera()
        
    elif menu == "База данных":
        show_database()

if __name__ == "__main__":
    main()