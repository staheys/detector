# detector/tests/test_camera_stream.py
import pytest
import cv2
from smart_detector.camera_stream import start_camera_stream

def test_start_camera_stream_importable():
    assert callable(start_camera_stream)

def test_invalid_source(monkeypatch):
    # Создаем объект-заменитель без определения класса
    dummy_cap = lambda: None
    dummy_cap.isOpened = lambda: False
    dummy_cap.release = lambda: None

    # ВидеоЗахват всегда возвращает dummy_cap
    monkeypatch.setattr(cv2, 'VideoCapture', lambda src: dummy_cap)

    with pytest.raises(RuntimeError):
        start_camera_stream(source=999, reconnect=False)
