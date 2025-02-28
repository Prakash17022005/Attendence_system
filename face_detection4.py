import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime
import winsound
import csv

#File path
user_file="users.csv"
file_name="attendance.xlsx"

# Load known faces Known face encodings and details 
known_face_encodings = []
known_face_names = []
known_roll_numbers = {}

if os.path.exists(user_file):
    with open(user_file,newline='') as csvfile:
        reader=csv.reader(csvfile)
        next(reader)
        for row in reader:
            name,roll_number,image_path=row
            image=face_recognition.load_image_file(image_path)
            encoding=face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            known_roll_numbers[name]=roll_number
    
# Dictionary to track attendance (prevents duplicate entries)
marked_attendance = {}


# Start webcam
cap = cv2.VideoCapture(0)

# Get today's date for the sheet name
today_date = datetime.now().strftime("%Y-%m-%d")
file_name = "attendance.xlsx"

# Function to mark attendance
def mark_attendance(name):
    roll_number = known_roll_numbers.get(name, "Unknown")
    time_now = datetime.now().strftime("%H:%M:%S")

    if name not in marked_attendance:  # ✅ Use the correct dictionary
        marked_attendance[name] = time_now  # Store attendance time
        
        winsound.Beep(1000,200)

        if os.path.exists(file_name):
            with pd.ExcelWriter(file_name, mode='a', if_sheet_exists='overlay') as writer:
                try:
                    df = pd.read_excel(file_name, sheet_name=today_date)
                except:
                    df = pd.DataFrame(columns=['Name', 'Roll Number', 'Time'])

                # Add new entry if not already present
                new_entry = pd.DataFrame([[name, roll_number, time_now]], columns=['Name', 'Roll Number', 'Time'])
                df = pd.concat([df, new_entry], ignore_index=True)
                df.to_excel(writer, sheet_name=today_date, index=False)

        else:
            df = pd.DataFrame([[name, roll_number, time_now]], columns=['Name', 'Roll Number', 'Time'])
            with pd.ExcelWriter(file_name, mode='w') as writer:
                df.to_excel(writer, sheet_name=today_date, index=False)

# Face Recognition Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            mark_attendance(name)  # ✅ Marks attendance only once per session

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    y_offset=30
    for i,(attendee,time) in enumerate(marked_attendance.items()):
        text=f"{attendee} - {time}"
        cv2.putText(frame,text,(10,y_offset+(i*30)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    
    cv2.imshow('Face Recognition - Attendance', frame)

    # Exit conditions
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Face Recognition - Attendance', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()



#Real-Time User Registration (Enroll Faces Dynamically)
"""
def register_user(name, roll_number, image_path):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]

    with open("users.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, roll_number, image_path])

    print(f"✅ {name} Registered Successfully!")
"""

