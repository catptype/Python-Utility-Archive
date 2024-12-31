import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Tuple, Union

class Converter:
    """A utility class for converting between different image and data formats."""

    ### Bytes Converters ###
    @staticmethod
    def byte_to_base64(byte_data: bytes) -> Union[str, None]:
        """Convert bytes to a base64 string."""
        if byte_data is None:
            return None
        try:
            return base64.b64encode(byte_data).decode("utf-8")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid byte data provided: {e}")

    @staticmethod
    def byte_to_pil(byte_data: bytes) -> Union[Image.Image, None]:
        """Convert bytes to a PIL Image."""
        if byte_data is None:
            return None
        try:
            return Image.open(BytesIO(byte_data))
        except Exception as e:
            raise ValueError(f"Failed to convert bytes to PIL Image: {e}")

    @staticmethod
    def byte_to_cv2(byte_data: bytes) -> Union[np.ndarray, None]:
        """Convert bytes to an OpenCV image (NumPy array)."""
        pil_image = Converter.byte_to_pil(byte_data)
        return Converter.pil_to_cv2(pil_image)

    ### Base64 Converters ###
    @staticmethod
    def base64_to_byte(base64_string: str) -> Union[bytes, None]:
        """Convert a base64 string to bytes."""
        if base64_string is None:
            return None
        try:
            return base64.b64decode(base64_string)
        except (base64.binascii.Error, ValueError) as e:
            raise ValueError(f"Invalid base64 string: {e}")

    @staticmethod
    def base64_to_pil(base64_string: str) -> Union[Image.Image, None]:
        """Convert a base64 string to a PIL Image."""
        if base64_string is None:
            return None
        try:
            image_data = base64.b64decode(base64_string)
            return Image.open(BytesIO(image_data))
        except Exception as e:
            raise ValueError(f"Failed to convert base64 to PIL Image: {e}")

    @staticmethod
    def base64_to_cv2(base64_string: str) -> Union[np.ndarray, None]:
        """Convert a base64 string to an OpenCV image (NumPy array)."""
        pil_image = Converter.base64_to_pil(base64_string)
        return Converter.pil_to_cv2(pil_image)

    ### OpenCV Image Converters ###
    @staticmethod
    def cv2_to_base64(numpy_array: np.ndarray) -> str:
        """Convert an OpenCV image (NumPy array) to a base64 string."""
        _, buffer = cv2.imencode(".jpg", numpy_array)
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def cv2_to_byte(numpy_array: np.ndarray) -> bytes:
        """Convert an OpenCV image (NumPy array) to bytes."""
        return base64.b64decode(Converter.cv2_to_base64(numpy_array))

    @staticmethod
    def cv2_to_pil(numpy_array: np.ndarray) -> Image.Image:
        """Convert an OpenCV image (NumPy array) to a PIL Image."""
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        try:
            return Image.fromarray(cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB))
        except Exception as e:
            raise ValueError(f"Failed to convert OpenCV image to PIL Image: {e}")

    ### Pillow Image Converters ###
    @staticmethod
    def pil_to_base64(pil_image: Image.Image) -> str:
        """Convert a PIL Image to a base64 string."""
        return Converter.cv2_to_base64(Converter.pil_to_cv2(pil_image))

    @staticmethod
    def pil_to_byte(pil_image: Image.Image) -> bytes:
        """Convert a PIL Image to bytes."""
        return base64.b64decode(Converter.pil_to_base64(pil_image))

    @staticmethod
    def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        """Convert a PIL Image to an OpenCV image (NumPy array)."""
        if not isinstance(pil_image, Image.Image):
            raise TypeError("Input must be a PIL Image.")
        try:
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Failed to convert PIL Image to OpenCV image: {e}")
    
    ### YOLO Bounding Box Converters ###
    @staticmethod
    def xywh2xyxy(
        bbox: Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]
    ) -> Tuple[float, float, float, float]:
        """Convert bbox from (center_x, center_y, width, height) to (x1, y1, x2, y2)."""
        center_x, center_y, width, height = bbox
        x1 = center_x - (width / 2.0)
        y1 = center_y - (height / 2.0)
        x2 = center_x + (width / 2.0)
        y2 = center_y + (height / 2.0)
        return x1, y1, x2, y2

    @staticmethod
    def xyxy2xywh(
        bbox: Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]
    ) -> Tuple[float, float, float, float]:
        """Convert bbox from (x1, y1, x2, y2) to (center_x, center_y, width, height)."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        return center_x, center_y, width, height
    
    @staticmethod
    def xyxy_norm2scalar(
        bbox: Tuple[float, float, float, float], image_width: int, image_height: int
    ) -> Tuple[float, float, float, float]:
        """Scale normalized bbox coordinates (x1, y1, x2, y2) to image dimensions."""
        x1, y1, x2, y2 = bbox

        x1 *= image_width
        x2 *= image_width
        y1 *= image_height
        y2 *= image_height

        return x1, y1, x2, y2
    
    @staticmethod
    def keypoint_norm2scalar(
        keypoint: Union[Tuple[float, float], Tuple[float, float, int]], image_width: int, image_height: int,
    ) -> Tuple[float, float]:
        """Scale normalized keypoint coordinates to image dimensions."""
        if len(keypoint) not in [2, 3]:
            raise ValueError(f"Invalid keypoint format: {keypoint}. Expected [x, y] or [x, y, flag].")
        x = keypoint[0] * image_width
        y = keypoint[1] * image_height
        return x, y