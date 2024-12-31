import cv2
import threading
import time
import numpy as np

from ping3 import ping
from ping3.errors import PingError
from typing import NoReturn, Literal
from onvif import ONVIFCamera

CAMERA_RESOLUTION = {
    '4K':   { 'width': 3840, 'height': 2160 },
    'QHD':  { 'width': 2560, 'height': 1440 },
    'FHD':  { 'width': 1920, 'height': 1080 },
    'HD':   { 'width': 1280, 'height': 720  },
    'SD':   { 'width': 640,  'height': 360  },
}
 
USE_DEMO_VIDEO = False # Remove this later

def get_camera_mac_address(ip:str, onvif_port:int, username:str, password:str) -> str:
    # Connect to the camera
    camera = ONVIFCamera(ip, onvif_port, username, password)
    
    # Access the device information
    devicemgmt_service = camera.create_devicemgmt_service()
    device_info = devicemgmt_service.GetDeviceInformation()
    
    # Retrieve and print the serial number or MAC address
    mac_address = device_info.SerialNumber

    return mac_address

class VideoFile:
    """
    A class to manage video file playback and retrieve video properties.

    Attributes:
        path (str): The path to the video file.
    """

    def __init__(self, path: str):
        """
        Initialize the VideoFile object.

        Args:
            path (str): Path to the video file.
        """
        self.__video = cv2.VideoCapture(path)
        self.__frame = None

    def __update(self):
        """
        Update the current frame by reading from the video file.

        Raises:
            ValueError: If the video file is not opened.
        """
        if self.__video.isOpened():
            ret, frame = self.__video.read()
            if ret:
                self.__frame = frame
        else:
            raise ValueError("Video is not opened")

    def stop(self) -> None:
        """
        Stop the video file and release resources.
        """
        print("Stop video file!!")
        self.__frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.__video.release()

    @property
    def frame(self) -> np.ndarray:
        """
        Get the current frame of the video.

        Returns:
            np.ndarray: The current frame as a NumPy array.
        """
        self.__update()
        return self.__frame

    @property
    def width(self) -> int:
        """
        Get the width of the video frame.

        Returns:
            int: The width of the video frame.
        """
        return int(self.__video.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """
        Get the height of the video frame.

        Returns:
            int: The height of the video frame.
        """
        return int(self.__video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> int:
        """
        Get the frames per second (FPS) of the video.

        Returns:
            int: The FPS of the video.
        """
        return int(self.__video.get(cv2.CAP_PROP_FPS))

    @property
    def num_frame(self) -> int:
        """
        Get the total number of frames in the video.

        Returns:
            int: The total number of frames.
        """
        return int(self.__video.get(cv2.CAP_PROP_FRAME_COUNT))
    
class UsbCamera:
    """
    A class to manage USB camera streams with support for configurable resolutions, frame rates,
    and automatic reconnection.

    Attributes:
        usb_id (int): The ID of the USB camera (default is 0).
        preset (Literal['SD', 'HD', 'FHD', 'QHD', '4K']): The resolution preset for the camera (default is 'HD').
        fps (int): The frame rate for the camera (default is 30).
    """
    def __init__(self, usb_id:int=0, preset:Literal['SD', 'HD', 'FHD', 'QHD', '4K']='HD', fps:int=30) -> None:
        """
        Initialize the USB camera object.

        Args:
            usb_id (int): The ID of the USB camera (default is 0).
            preset (Literal['SD', 'HD', 'FHD', 'QHD', '4K']): The resolution preset for the camera (default is 'HD').
            fps (int): The frame rate for the camera (default is 30).
        """
        self.__usb_id = usb_id
        self.__preset = preset
        self.__fps = fps
        self.__camera = self.__set_camera_properties()
        self.__ret, self.__frame = self.__camera.read()
        self.__is_activated = True

        self.__thread = threading.Thread(target=self.__update, daemon=True)
        self.__thread.start()

    def __update(self) -> NoReturn:
        """
        Continuously read frames from the camera in a separate thread.

        If the camera is not working, attempts to reconnect automatically.
        """
        while self.__is_activated:
            if not self.__camera.isOpened() or not self.__ret:
                print("Camera is not working.  Attempting to reconnect ...")
                self.__reconnect()
            else:
                self.__ret, self.__frame = self.__camera.read()

    def __reconnect(self) -> None:
        """
        Attempt to reconnect to the camera.

        Releases the current camera resource, waits for 3 seconds, and reinitializes it.
        """
        print("Camera is reconnecting ...")
        self.__frame = np.zeros((self.__height, self.__width, 3), dtype=np.uint8)
        self.__camera.release()
        time.sleep(3)
        self.__camera = self.__set_camera_properties()
        self.__ret, self.__frame = self.__camera.read()

    def __set_camera_properties(self):
        """
        Configure the camera with the specified properties.

        Returns:
            cv2.VideoCapture: The configured camera object.
        """
        camera = cv2.VideoCapture(self.__usb_id)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[self.__preset]['width'])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[self.__preset]['height'])
        camera.set(cv2.CAP_PROP_FPS, self.__fps)
        return camera
    
    def stop(self) -> None:
        """
        Stop the camera stream and release resources.
        """
        print("Stop camera!!")
        self.__is_activated = False
        self.__frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.__thread.join()
        self.__camera.release()        
    
    @property
    def frame(self) -> np.ndarray:
        """
        Get the latest frame captured by the camera.

        Returns:
            np.ndarray: The latest frame as a NumPy array.
        """
        return self.__frame

    @property
    def width(self) -> int:
        """
        Get the width of the camera's resolution.

        Returns:
            int: The width of the camera's resolution.
        """
        return int(self.__camera.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """
        Get the height of the camera's resolution.

        Returns:
            int: The height of the camera's resolution.
        """
        return int(self.__camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> int:
        """
        Get the current frame rate of the camera.

        Returns:
            int: The current frame rate.
        """
        return self.__camera.get(cv2.CAP_PROP_FPS)

class RtspCamera:
    def __init__(self, url:str, ip:str, rtsp_port:int, username:str, password:str, suffix:str, onvif_port:int) -> None:
        self.__ip = ip
        self.__rtsp_port = rtsp_port
        self.__username = username
        self.__password = password
        self.__onvif_port = onvif_port
        self.__url_suffix = suffix
        self.__url = f"rtsp://{self.__username}:{self.__password}@{self.__ip}:{self.__rtsp_port}{self.__url_suffix}" if url is None else url  
        self.__camera = cv2.VideoCapture(self.__url)
        self.__ret, self.__frame = self.__camera.read()

        try:
            self.__mac_address = get_camera_mac_address(self.__ip, self.__onvif_port, self.__username, self.__password)
        except:
            self.__mac_address = "FFFFFFFFFFFF"

        self.__width = int(self.__camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.__height = int(self.__camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__fps = int(self.__camera.get(cv2.CAP_PROP_FPS))
        self.__is_activated = True
        self.__is_online = False 
        self.__ping_time = -1
        self.__frame_duration = 1.0 / self.__fps

        self.__thread = threading.Thread(target=self.__update, daemon=True)
        self.__thread.start()

        self.__ping_thread = threading.Thread(target=self.__ping, daemon=True)
        self.__ping_thread.start()
    
    def __del__(self) -> None:
        self.stop()
    
    def __ping(self) -> NoReturn:
        while self.__is_activated:
            try:
                ping_result = ping(self.ip, timeout=4, unit="ms")
                if ping_result:
                    self.__is_online = True
                    self.__ping_time = int(ping_result)
                else:
                    self.__is_online = False
                    self.__ping_time = -1
            except PingError:
                self.__is_online = False
                self.__ping_time = -1
            
            time.sleep(5)
        
        self.__ping_time = -1
    
    def __get_address_component(self, index: int) -> str:
        """
        rtsp url patterns
        1. "rtsp://<user>:<password>@<ip>:<port>/streaming/channels/101/" # HIKVISION
        2. "rtsp://<user>:<password>@<ip>:<port>" # REOLINK
        3. "rtsp://<user>:<password>@<ip>:<port>/1" # APOLLO
        """
        if not 'rtsp://' in str(self.__url):
            return "localhost"
        
        url = self.__url.replace('rtsp://', '').split('/')[0]
        address = url.split('@')[-1]
        return address.split(':')[index]

    def __attempt_read(self) -> bool:
        print("Camera connection is unstable. Attempting to retrieve video frame 5 times ...")
        for _ in range(5):
            start_time = time.time()
            self.__ret, temp_frame = self.__camera.read()
            if self.__ret:
                self.__frame = temp_frame
                return True
            elapsed_time = time.time() - start_time
            sleep_time = self.__frame_duration - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        return False

    def __update(self) -> NoReturn:
        """
        Temporary use this class object for video file.
        In future, I will write class object for video file separately
        """
        if USE_DEMO_VIDEO:
            while self.__is_activated:
                if not self.__camera.isOpened() or not self.__ret:                
                    if self.__attempt_read():
                        continue
                    print("Camera is disconnected. Attempting to reconnect ...")
                    self.__reconnect()
                else:
                    # Must calculate processing time for sleeping
                    start_time = time.time()
                    self.__ret, self.__frame = self.__camera.read()
                    elapsed_time = time.time() - start_time
                    sleep_time = self.__frame_duration - elapsed_time
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        else:
            while self.__is_activated:
                if not self.__camera.isOpened() or not self.__ret:                
                    if self.__attempt_read():
                        continue
                    print("Camera is disconnected. Attempting to reconnect ...")
                    self.__reconnect()
                else:
                    self.__ret, self.__frame = self.__camera.read()
    
    def __reconnect(self) -> None:
        print("Camera is reconnecting ...")
        self.__frame = np.zeros((self.__height, self.__width, 3), dtype=np.uint8)
        self.__camera.release()
        time.sleep(3)
        self.__camera = cv2.VideoCapture(self.__url)
        self.__ret, self.__frame = self.__camera.read()
        try:
            self.__mac_address = get_camera_mac_address(self.__ip, self.__onvif_port, self.__username, self.__password)
        except:
            self.__mac_address = "FFFFFFFFFFFF"
        self.__width = int(self.__camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.__height = int(self.__camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__fps = int(self.__camera.get(cv2.CAP_PROP_FPS))
    
    def stop(self) -> None:
        if self.__is_activated:
            print("Camera is stopping ...")
            self.__is_activated = False
            self.__is_online = False
            self.__frame = np.zeros((self.__height, self.__width, 3), dtype=np.uint8)
            self.__camera.release()
            self.__thread.join()  # Ensure the thread is fully terminated
            self.__ping_thread.join()
            print("Camera is stopped")

    def start(self) -> None:
        if self.__is_activated:
            print("Camera is already activated")
        else:
            print("Start camera in 3 seconds")
            time.sleep(3)
            self.__camera = cv2.VideoCapture(self.__url)
            self.__ret, self.__frame = self.__camera.read()
            self.__width = int(self.__camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.__height = int(self.__camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.__fps = round(self.__camera.get(cv2.CAP_PROP_FPS), 2)
            self.__is_activated = True
            self.__is_online = False 
            self.__ping_time = -1

            self.__thread = threading.Thread(target=self.__update, daemon=True)
            self.__thread.start()
            
            self.__ping_thread = threading.Thread(target=self.__ping, daemon=True)
            self.__ping_thread.start()
    
    def is_offline(self) -> bool:
        return not self.__is_online
    
    def is_connecting(self) -> bool:
        return not self.__camera.isOpened()
    
    @property
    def frame(self) -> np.ndarray:
        return self.__frame
    
    @property
    def ping_time(self) -> int:
        return self.__ping_time

    @property
    def ip(self) -> str:
        return self.__get_address_component(0)

    @property
    def port(self) -> str:
        return self.__get_address_component(1)

    @property
    def width(self) -> int:
        return self.__width
    
    @property
    def height(self) -> int:
        return self.__height

    @property
    def fps(self) -> int:
        return self.__fps

    @property
    def mac_address(self) -> str:
        return self.__mac_address