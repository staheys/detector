# src/smart_detector/camera_stream.py
"""
Модуль для захвата видеопотока, распознавания объектов с помощью Ultralytics YOLO
и отображения результата в окне, размер которого соответствует потоку.
"""
import cv2
import configparser
import os
import time
import logging
from typing import Union
from ultralytics import YOLO

def start_camera_stream(
    source: Union[int, str],
    model: YOLO,
    conf_threshold: float,
    window_name: str = None,
    reconnect: bool = False,
    reconnect_interval: float = 5.0
):
    """
    Запускает захват видео из источника, выполняет детекцию объектов
    и показывает кадры с наложенными прямоугольниками.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    logger = logging.getLogger(__name__)
    target = f"RTSP-поток '{source}'" if isinstance(source, str) else f"камера индекс {source}"
    logger.info(f"Открываем {target}...")

    # Запуск потока обработки окон (для Windows GUI)
    cv2.startWindowThread()

    while True:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error(f"Не удалось открыть источник: {target}.")
            cap.release()
            if isinstance(source, str) and reconnect:
                logger.info(f"Переподключение через {reconnect_interval} сек...")
                time.sleep(reconnect_interval)
                continue
            raise RuntimeError(f"Не удалось открыть видеоисточник: {source}")

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            logger.warning("Не удалось установить буфер кадров, используем стандартный.")

        try:
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise ValueError(f"Недопустимые размеры кадра: {width}x{height}")

            win_name = window_name or f"{source} - {width}x{height}"
            # Используем автоподгонку размера окна
            cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
            # Один раз прогон для обработки события открытия
            cv2.waitKey(1)
        except Exception:
            cap.release()
            logger.exception("Ошибка при настройке окна.")
            raise

        try:
            while True:
                if isinstance(source, str):
                    # Обнуляем буфер RTSP
                    while cap.grab():
                        pass

                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("Кадр не получен или пустой.")
                    break

                results   = model(frame, conf=conf_threshold)[0]
                annotated = results.plot()

                cv2.imshow(win_name, annotated)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    logger.info("Сигнал выхода получен, закрываем.")
                    return
        except KeyboardInterrupt:
            logger.info("Остановлено пользователем.")
            return
        except Exception:
            logger.exception("Ошибка во время стрима.")
            raise
        finally:
            cap.release()
            cv2.destroyWindow(win_name)

        if isinstance(source, str) and reconnect:
            logger.info(f"Поток прерван, переподключение через {reconnect_interval} сек...")
            time.sleep(reconnect_interval)
            continue
        break

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stream + YOLO-детекцию")
    parser.add_argument(
        "-c", "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), 'config.ini'),
        help="Путь к INI-файлу конфигурации"
    )
    args = parser.parse_args()

    cfg = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
    if not os.path.exists(args.config):
        parser.error(f"Файл конфигурации не найден: {args.config}")
    cfg.read(args.config)

    # Параметры потока
    src        = cfg.get('stream', 'source')
    name       = cfg.get('stream', 'name', fallback=None)
    reconnect  = cfg.getboolean('stream', 'reconnect', fallback=False)
    interval   = cfg.getfloat('stream', 'interval', fallback=5.0)

    # Параметры детекции
    weights    = cfg.get('detection', 'weights', fallback='yolov8n.pt')
    device     = cfg.get('detection', 'device', fallback='cpu')
    conf_thresh= cfg.getfloat('detection', 'confidence', fallback=0.25)

    # Интерпретация источника
    if src.isdigit():
        source = int(src)
    else:
        source = src

    # Загрузка модели
    model = YOLO(weights)
    model.to(device)

    start_camera_stream(
        source=source,
        model=model,
        conf_threshold=conf_thresh,
        window_name=name,
        reconnect=reconnect,
        reconnect_interval=interval
    )

if __name__ == "__main__":
    main()
