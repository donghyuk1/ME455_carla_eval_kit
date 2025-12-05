import zmq
import numpy as np
import cv2
import sys
import time

# --- ZMQ Configuration (Must match the publisher) ---
ZMQ_PORT = "5555"
SERVER_ADDRESS = f"tcp://127.0.0.1:{ZMQ_PORT}"
TOPIC = b"HDMAP"

def receive_and_display():
    """
    Connects to the ZMQ publisher, receives image data, and displays it via OpenCV.
    """
    # 1. Setup ZMQ Subscriber
    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    print(f"Connecting to HDMap Publisher at {SERVER_ADDRESS}...")
    try:
        # Use 127.0.0.1 to connect locally.
        socket.connect(SERVER_ADDRESS)
    except zmq.error.ZMQError as e:
        print(f"Error connecting to ZMQ socket: {e}")
        print("Ensure hdmap.py (Publisher) is running.")
        context.term()
        sys.exit(1)
    
    # Subscribe to the specific topic (TOPIC + space)
    socket.setsockopt(zmq.SUBSCRIBE, TOPIC)

    # 2. Setup OpenCV Display Window (Runs safely on this thread)
    WINDOW_NAME = "HDMap Real-Time Viewer (ZMQ)"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 800)
    
    # Optional: Set a timeout to check for server startup/disconnect
    socket.setsockopt(zmq.RCVTIMEO, 2000) # Timeout after 2000ms

    print("Successfully subscribed. Waiting for first image...")
    
    try:
        while True:
            # 3. Receive Message
            try:
                message = socket.recv()
            except zmq.error.Again:
                # Timeout occurred, server might be slow or disconnected
                print("Waiting for data (timeout)...", end='\r')
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                continue
                
            # 4. Process Message (Topic and Data)
            if message.startswith(TOPIC + b" "):
                # Extract the raw JPEG data (everything after "TOPIC ")
                jpeg_data = message[len(TOPIC) + 1:]
                
                if not jpeg_data:
                    continue

                # 5. Decode Image (from JPEG bytes back to NumPy array)
                # This uses the same CV2 library that was forbidden in the publisher,
                # but is safe here because it runs on this thread.
                nparr = np.frombuffer(jpeg_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # 6. Display
                if img is not None:
                    cv2.imshow(WINDOW_NAME, img)
                    
            # CRITICAL: This processes the display events and handles the 'q' keypress.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nViewer manually shut down.")
    finally:
        cv2.destroyAllWindows()
        context.term()

if __name__ == '__main__':
    receive_and_display()