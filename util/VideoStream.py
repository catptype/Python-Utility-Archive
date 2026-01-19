import cv2
import threading
import time
from typing import Union, Optional

class VideoStream:
    """
    Threaded VideoCapture for RTSP, Webcams, and Video Files.
    Always runs in a background thread to ensure the main application 
    gets the most recent frame instantly without I/O blocking.
    """

    def __init__(self, source: Union[int, str], name: str = "Stream"):
        self.source = source
        self.name = name
        self.stopped = False
        self._frame = None
        
        # Threading Lock
        self.lock = threading.Lock()

        # 1. Initialize Capture
        self.cap = cv2.VideoCapture(self.source)
        
        # Optimization for RTSP: reduce buffer to keep it 'live'
        if isinstance(source, str) and (source.startswith("rtsp") or source.startswith("http")):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise ValueError(f"[{self.name}] Could not open source: {self.source}")

        # 2. Cache Properties (Avoids locking issues later)
        # We read these once so accessing stream.width is instant and lock-free.
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 3. Read first frame to ensure validity
        ret, frame = self.cap.read()
        if ret:
            self._frame = frame
        else:
            print(f"[{self.name}] Warning: Source opened but returned no frame.")

        # 4. Start Background Thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """
        Background thread loop. 
        Crucial: cap.read() is done OUTSIDE the lock to prevent blocking.
        """
        while not self.stopped:
            if not self.cap.isOpened():
                self.stop()
                break

            # BLOCKING I/O operation happens here (unlocked)
            # This allows the main program to access self.frame even while the camera is reading.
            ret, frame = self.cap.read()

            if not ret:
                # End of stream (video file finished or network lost)
                self.stop()
                break

            # MEMORY operation happens here (locked, extremely fast)
            with self.lock:
                self._frame = frame
            
            # If playing a video file, we might want to sleep slightly to match FPS
            # Otherwise, the thread reads the whole file in 1 second.
            if self.total_frames > 0 and self.fps > 0:
                time.sleep(1 / self.fps)

    @property
    def frame(self) -> Optional[cv2.Mat]:
        """
        Property to get the latest frame.
        Usage: img = stream.frame
        Returns: cv2.Mat or None if stream ended.
        """
        with self.lock:
            if self._frame is None:
                return None
            # Return a copy to ensure thread safety 
            # (prevents main thread from modifying image while update thread overwrites it)
            return self._frame.copy()

    @property
    def is_running(self) -> bool:
        """Check if stream is still active."""
        return not self.stopped and self.cap.isOpened()

    @property
    def info(self) -> dict:
        """Returns metadata about the stream (Cached, no locking needed)."""
        return {
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "status": "Running" if self.is_running else "Stopped"
        }

    def stop(self):
        """Stops the thread and releases resources."""
        self.stopped = True
        # Wait a moment for thread to finish
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        
        if self.cap.isOpened():
            self.cap.release()
            
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()