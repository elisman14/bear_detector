import streamlit as st
from PIL import Image
import numpy as np
import torch
import sys
import os
import cv2
import platform
import pathlib



if platform.system() == "Windows":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pass


yolov5_path = os.path.join(os.path.dirname(__file__), 'yolov5')

if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
    from utils.torch_utils import select_device
    from utils.augmentations import letterbox
    from utils.plots import Annotator, colors
except ImportError as e:
    st.error(
        f"Не удалось импортировать модули: {e}")
    st.stop()

# --- Конфигурация ---
MODEL_WEIGHTS_PATH = 'best.pt'
IMG_SIZE = (640, 640)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


# --- Функции ---

@st.cache_resource
def load_yolov5_model(weights_path=MODEL_WEIGHTS_PATH):
    try:
        device = select_device('')  # '' для автоматического выбора (CPU/GPU)
        model = DetectMultiBackend(weights_path, device=device, dnn=False, data=None, fp16=False)
        model.eval()  # Переводим модель в режим оценки
        st.success("Успешно!")
        return model, device
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        st.stop()


def preprocess_image(image_pil, img_size, stride):

    img_np = np.array(image_pil)
    img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    img = letterbox(img_np_bgr, img_size, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    return img_np_bgr, img


def run_inference(model, device, preprocessed_img_tensor):

    preprocessed_img_tensor = torch.from_numpy(preprocessed_img_tensor).to(device)
    preprocessed_img_tensor = preprocessed_img_tensor.float()  # uint8 to fp16/32
    preprocessed_img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
    if preprocessed_img_tensor.ndimension() == 3:
        preprocessed_img_tensor = preprocessed_img_tensor.unsqueeze(0)  # Добавляем размерность пакета

    pred = model(preprocessed_img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD, classes=None, agnostic=False,
                               max_det=1000)
    return pred, preprocessed_img_tensor


def draw_detections(original_img_bgr, detections, img_shape_for_scaling, names):

    annotator = Annotator(original_img_bgr.copy(), line_width=2, example=str(names))
    detections_found = False

    for i, det in enumerate(detections):
        if len(det):
            detections_found = True
            det[:, :4] = scale_boxes(img_shape_for_scaling, det[:, :4], original_img_bgr.shape).round()

            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()
                st.write(f"- Обнаружен {int(n)} Медведей")

            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]}'
                annotator.box_label(xyxy, label, color=colors(int(cls), True))

    result_img = annotator.result()
    return result_img, detections_found


def main():
    """
    Главная функция Streamlit приложения.
    """
    st.set_page_config(page_title="Детектирование медведей вблизи населенных пунктов.", layout="centered")

    st.title("Детектирование медведей вблизи населенных пунктов.")
    st.write("Загрузите изображение, чтобы обнаружить на нем медведей.")

    # 1. Загрузка модели
    model, device = load_yolov5_model()
    stride, names, pt = model.stride, model.names, model.pt  # Получаем параметры модели

    # 2. Загрузка файла пользователем
    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Отображаем загруженное изображение
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_container_width=True)
        st.write("")
        st.write("Выполняется предсказание...")

        # 3. Предобработка изображения
        original_img_bgr, preprocessed_img_tensor = preprocess_image(image, IMG_SIZE, stride)

        # 4. Выполнение инференса
        pred, img_shape_for_scaling = run_inference(model, device, preprocessed_img_tensor)

        # 5. Отображение результатов
        result_img, detections_found = draw_detections(original_img_bgr, pred, img_shape_for_scaling.shape[2:], names)

        if detections_found:
            st.image(result_img, caption="Результат обнаружения", use_container_width=True)
        else:
            st.write("Медведи не обнаружены на изображении.")


# Запуск приложения
if __name__ == "__main__":
    main()