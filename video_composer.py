# video_composer.py
import cv2
import os
import datetime
import logging
import tempfile  # Для создания временных файлов
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def format_timedelta_for_ffmpeg(td: datetime.timedelta) -> str:
    """Форматирует timedelta в строку HH:MM:SS.mmm для ffmpeg."""
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def create_composite_video_opencv(
        video_segments: List[Dict],
        output_filename: str,
        recordings_path: str,
        target_fps: Optional[float] = None,  # FPS для выходного файла
        target_width: Optional[int] = None,  # Ширина для выходного файла
        target_height: Optional[int] = None  # Высота для выходного файла
) -> Optional[str]:
    """
    Создает композитное видео из указанных сегментов с использованием OpenCV.

    Args:
        video_segments: Список словарей, где каждый словарь содержит:
                        'filename': имя исходного видеофайла.
                        'start_offset_in_clip': timedelta, смещение начала нужного фрагмента от начала клипа.
                        'end_offset_in_clip': timedelta, смещение конца нужного фрагмента от начала клипа.
        output_filename: Имя выходного файла (без пути).
        recordings_path: Путь к директории с исходными видео.
        target_fps: Желаемый FPS для выходного видео. Если None, используется FPS первого клипа.
        target_width: Желаемая ширина. Если None, используется из первого клипа.
        target_height: Желаемая высота. Если None, используется из первого клипа.

    Returns:
        Полный путь к созданному файлу или None в случае ошибки.
    """
    if not video_segments:
        logger.warning("Нет сегментов для компоновки видео.")
        return None

    # Определяем путь для временного выходного файла
    # temp_dir = tempfile.gettempdir() # Можно использовать системный temp
    # output_path = os.path.join(temp_dir, output_filename)
    # Или сохранять рядом с записями, но с префиксом temp_
    temp_output_dir = os.path.join(recordings_path, "composites_temp")
    os.makedirs(temp_output_dir, exist_ok=True)
    output_path = os.path.join(temp_output_dir, output_filename)

    writer = None
    final_width, final_height, final_fps = target_width, target_height, target_fps

    total_frames_written = 0

    try:
        for i, segment_info in enumerate(video_segments):
            clip_filename = segment_info['filename']
            start_offset_td = segment_info['start_offset_in_clip']
            end_offset_td = segment_info['end_offset_in_clip']

            clip_path = os.path.join(recordings_path, clip_filename)
            if not os.path.exists(clip_path):
                logger.error(f"Файл клипа не найден: {clip_path}")
                continue

            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened():
                logger.error(f"Не удалось открыть клип: {clip_path}")
                continue

            current_clip_fps = cap.get(cv2.CAP_PROP_FPS)
            current_clip_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            current_clip_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if i == 0:  # Инициализация writer на основе первого клипа, если параметры не заданы
                if final_fps is None: final_fps = current_clip_fps
                if final_width is None: final_width = current_clip_width
                if final_height is None: final_height = current_clip_height

                if final_fps <= 0:  # Если FPS все еще невалидный
                    logger.warning(f"FPS первого клипа {current_clip_fps} невалиден. Используем 25.0 по умолчанию.")
                    final_fps = 25.0

                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # или другой кодек, например, MJPG
                writer = cv2.VideoWriter(output_path, fourcc, final_fps, (final_width, final_height))
                if not writer.isOpened():
                    logger.error(f"Не удалось инициализировать VideoWriter для {output_path}")
                    cap.release()
                    return None
                logger.info(
                    f"Композитное видео будет создано: {output_path} с {final_fps} FPS, {final_width}x{final_height}")

            start_frame_num = int(start_offset_td.total_seconds() * current_clip_fps)
            end_frame_num = int(end_offset_td.total_seconds() * current_clip_fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)
            frames_in_segment = 0

            logger.info(f"Обработка сегмента: {clip_filename}, кадры с {start_frame_num} по {end_frame_num}")

            for frame_count in range(start_frame_num, end_frame_num + 1):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Не удалось прочитать кадр {frame_count} из {clip_path}. Прерываем сегмент.")
                    break

                # Приведение кадра к целевому разрешению, если необходимо
                if frame.shape[1] != final_width or frame.shape[0] != final_height:
                    frame = cv2.resize(frame, (final_width, final_height), interpolation=cv2.INTER_AREA)

                writer.write(frame)
                frames_in_segment += 1

            logger.info(f"Добавлено {frames_in_segment} кадров из сегмента {clip_filename}.")
            total_frames_written += frames_in_segment
            cap.release()

        if writer:
            writer.release()

        if total_frames_written > 0:
            logger.info(f"Композитное видео успешно создано: {output_path}, всего кадров: {total_frames_written}")
            return output_path
        else:
            logger.warning(
                f"Композитное видео не было создано, так как не было записано ни одного кадра: {output_path}")
            if os.path.exists(output_path): os.remove(output_path)  # Удалить пустой файл
            return None

    except Exception as e:
        logger.exception(f"Ошибка во время создания композитного видео: {e}")
        if writer:
            writer.release()
        if os.path.exists(output_path) and total_frames_written == 0:  # Удалить если пустой или ошибка
            os.remove(output_path)
        return None