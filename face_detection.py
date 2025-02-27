import cv2  # Import OpenCV library

# Start the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Load Haar Cascade face detection model
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    print("❌ Error: Haar Cascade file not found.")
    exit()

print("✅ Webcam & Face Detection Model Loaded Successfully!")

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    
    if not ret:
        print("❌ Error: Failed to capture frame. Exiting...")
        break  # Exit if no frame is captured

    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Detection - Live(press q to exit)', frame)  # Show the video frame

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🔴 Exiting program...")
        break

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close the OpenCV window
