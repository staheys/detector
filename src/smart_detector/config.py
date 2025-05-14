# src/smart_detector/config.py
from pathlib import Path

# These are fallback defaults if not found in YAML or for constants
# Most settings will come from config.yml via utils.get_config()

# Display settings (primarily for cv2.imshow placeholder)
CV2_FONT = 0 # cv2.FONT_HERSHEY_SIMPLEX
CV2_FONT_SCALE_SMALL = 0.5
CV2_FONT_SCALE_LARGE = 0.7
CV2_FONT_THICKNESS = 1
CV2_LINE_TYPE = 16 # cv2.LINE_AA
CV2_COLOR_WHITE = (255, 255, 255)
CV2_COLOR_BLACK = (0, 0, 0)
CV2_COLOR_FPS = (255, 255, 0) # Cyan for FPS
CV2_FPS_POSITION = (10, 30)
CV2_EVENT_NOTIFICATION_COLOR_BG = (0, 0, 0, 180) # Black, semi-transparent for BG

# Default model name if not specified (though YAML should always specify model_path)
DEFAULT_MODEL_NAME = "yolov9-medium.onnx"

# Placeholder for application state/shared resources if needed without classes
# e.g. model_instance, zmq_context
# However, it's cleaner to pass these as arguments or manage within specific modules/threads

APP_NAME = "SmartDetector"