import cv2
import face_recognition
import numpy as np

known_face_encodings =[]
known_face_names =[]

image_1=face_recognition.load_image_file("images/prakash_img.jpg")
encoding_1=face_recognition.face_encodings(image_1)[0]
known_face_encodings.append(encoding_1)
known_face_names.append("Prakash")

image_2=face_recognition.load_image_file("images/joycy_img.jpg")
encoding_2=face_recognition.face_encodings(image_2)[0]
known_face_encodings.append(encoding_2)
known_face_names.append("mary")

image_3=face_recognition.load_image_file("images/kolass_img.jpg")
encoding_3=face_recognition.face_encodings(image_3)[0]
known_face_encodings.append(encoding_3)
known_face_names.append("kolass")

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame.")
        break
    
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    face_locations=face_recognition.face_locations(rgb_frame)
    face_encodings=face_recognition.face_encodings(rgb_frame,face_locations)
    
    for(top,right,bottom,left),face_encoding in zip(face_locations,face_encodings):
        matches=face_recognition.compare_faces(known_face_encodings,face_encoding)
        name="unknown"
        
        face_distances=face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index=np.argmin(face_distances)
        
        if matches[best_match_index]:
            name=known_face_names[best_match_index]
            
        cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2) 
        cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        
    cv2.imshow("Face Recognition - Live",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Face Recognition - Live',cv2.WND_PROP_VISIBLE)<1:
        break
    
cap.release()
cv2.destroyAllWindows()