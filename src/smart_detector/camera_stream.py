# detector/src/smart_detector/camera_stream.py
"""
Модуль для захвата видеопотока с камеры по индексу или RTSP-URL и отображения его в окне
с разрешением, соответствующим потоку, с обработкой ошибок и возможностью переподключения для RTSP.
"""
import cv2
import logging
import time
import os
import configparser
from typing import Union

def start_camera_stream(
    source: Union[int, str],
    window_name: str = None,
    reconnect: bool = False,
    reconnect_interval: float = 5.0
):
    """
    Запускает захват видео с камеры (индекс) или RTSP-URL и отображает его в окне
    с размером, соответствующим разрешению кадра. Поддерживает обработку ошибок
    и опциональное переподключение.

    :param source: индекс камеры (int) или RTSP-URL (str).
    :param window_name: пользовательское имя окна (необязательно).
    :param reconnect: флаг переподключения при обрыве потока (только для RTSP).
    :param reconnect_interval: время ожидания перед попыткой переподключения (секунды).
    """
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    logger = logging.getLogger(__name__)
    target = f"RTSP-поток '{source}'" if isinstance(source, str) else f"камера индекс {source}"
    logger.info(f"Открываем {target}...")

    # Основной цикл: попытка захвата и отображения
    while True:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Не удалось открыть источник: {target}.")
            cap.release()
            if isinstance(source, str) and reconnect:
                logger.info(f"Переподключение через {reconnect_interval} секунд...")
                time.sleep(reconnect_interval)
                continue
            raise RuntimeError(f"Не удалось открыть видеоисточник: {source}")

        # Настройка окна по размеру потока
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise ValueError(f"Недопустимые размеры кадра: {width}x{height}")

            final_window = window_name or f"{source} - {width}x{height}"
            cv2.namedWindow(final_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(final_window, width, height)
        except Exception:
            cap.release()
            logger.exception("Ошибка при настройке окна отображения.")
            raise

        # Цикл чтения и показа кадров
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("Не удалось получить кадр или кадр пустой.")
                    break

                cv2.imshow(final_window, frame)
                key = cv2.waitKey(1)
                if key & 0xFF in (ord('q'), 27):
                    logger.info("Получен сигнал выхода, закрываем поток.")
                    return
        except KeyboardInterrupt:
            logger.info("Стрим остановлен пользователем.")
            return
        except Exception:
            logger.exception("Непредвиденная ошибка во время стрима.")
            raise
        finally:
            cap.release()
            cv2.destroyWindow(final_window)

        # При необходимости переподключиться (RTSP)
        if isinstance(source, str) and reconnect:
            logger.info(f"Поток прерван. Переподключение через {reconnect_interval} секунд...")
            time.sleep(reconnect_interval)
            continue
        break

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Просмотр видеопотока с камеры или RTSP")
    parser.add_argument(
        "-c", "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), 'config.ini'),
        help="Путь к конфигурационному файлу INI"
    )
    parser.add_argument("-s", "--source", type=str, help="Индекс камеры или RTSP-URL")
    parser.add_argument("-n", "--name",   type=str, help="Пользовательское название окна")
    parser.add_argument("-r", "--reconnect", action="store_true", help="Включить переподключение для RTSP")
    parser.add_argument("-i", "--interval", type=float, help="Интервал переподключения в секундах")
    args = parser.parse_args()

    # Чтение конфигурации из INI
    config = configparser.ConfigParser()
    if os.path.exists(args.config):
        config.read(args.config)
        if 'stream' in config:
            sec = config['stream']
            if not args.source and sec.get('source'):
                args.source = sec.get('source')
            if not args.name   and sec.get('name'):
                args.name = sec.get('name')
            if not args.reconnect and sec.get('reconnect'):
                args.reconnect = sec.getboolean('reconnect')
            if args.interval is None and sec.get('interval'):
                args.interval = sec.getfloat('interval')

    if not args.source:
        parser.error("Необходимо указать источник видео через --source или в файле конфигурации.")

    # Преобразование источника
    if str(args.source).isdigit():
        source = int(args.source)
    else:
        source = args.source

    start_camera_stream(
        source,
        window_name=args.name,
        reconnect=args.reconnect,
        reconnect_interval=args.interval if args.interval is not None else 5.0
    )

if __name__ == "__main__":
    main()
