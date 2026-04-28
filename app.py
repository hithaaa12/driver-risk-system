import cv2
import mediapipe as mp
from drowsiness import eye_aspect_ratio

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33,160,158,133,153,144]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face in results.multi_face_landmarks:

            # Extract eye points
            eye_points = []

            h, w, _ = frame.shape

            for idx in LEFT_EYE:
                lm = face.landmark[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                eye_points.append((x, y))

            # Calculate EAR
            ear = eye_aspect_ratio(eye_points)

            print("EAR:", ear)

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2
            )

            # Simple drowsy flag
            if ear < 0.22:
                cv2.putText(
                    frame,
                    "DROWSY",
                    (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
                )

            # Draw facial landmarks
            for landmark in face.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(frame,(x,y),1,(0,255,0),-1)

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()