import cv2
from defect_detection_pipeline import detect_defects  # Make sure this function exists in your project

def main():
    # Open laptop webcam (0 is default)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit live defect detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Resize frame (optional for speed)
        frame = cv2.resize(frame, (640, 480))

        # Apply your defect detection pipeline here
        result = detect_defects(frame)

        # Show output in a window
        cv2.imshow("Live Bottle Defect Detection", result)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
