import cv2
from util.VideoStream import VideoStream

# Initialize
cam = VideoStream(0, name="Webcam")

print(f"Camera Resolution: {cam.width}x{cam.height}")
print(f"Camera Info {cam.info}")

while cam.is_running:
    # Get frame via property (No 'ret', no 'read()')
    image = cam.frame

    if image is not None:
        cv2.imshow("Live", image)
    
    if cv2.waitKey(1) == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()