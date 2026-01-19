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
        Uses Delta Time calculation to ensure video files play at correct speed.
        """
        # Pre-calculate target time per frame (e.g., 30 FPS = 0.0333s)
        target_frame_time = 1.0 / self.fps if self.fps > 0 else 0

        while not self.stopped:
            # 1. Start the timer for this iteration
            start_time = time.perf_counter()

            if not self.cap.isOpened():
                self.stop()
                break

            # 2. Blocking I/O (Read frame)
            ret, frame = self.cap.read()

            if not ret:
                self.stop()
                break

            # 3. Thread Safe Update
            with self.lock:
                self._frame = frame

            # 4. Smart Sleep (Synchronization)
            # Only apply strict timing if it is a video file (total_frames > 0)
            if self.total_frames > 0 and target_frame_time > 0:
                
                # Check how much time reading/locking actually took
                elapsed_time = time.perf_counter() - start_time
                
                # Calculate remaining time needed to hit target FPS
                time_to_wait = target_frame_time - elapsed_time

                if time_to_wait > 0:
                    time.sleep(time_to_wait)

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
            return self._frame

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