# Image Processing
import cv2  # For computer vision and image processing
import numpy as np  # Support for large, multi-dimensional arrays and matrices of numerical data and collection of mathematical functions
import mediapipe as mp  # type: ignore # For face mesh detection
import pandas as pd  # type: ignore # For creating a table of head movements
import matplotlib.pyplot as plt  #type: ignore For plotting graphs
from head_detection import detect_head  # Import the head detection function

def start_video_capture():
    cap = cv2.VideoCapture(0)  # Use the first camera device it finds

    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # List to store head movement data
    head_movements = []

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Error: Can't receive frame")
            break

        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get the result
        results = face_mesh.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calculate the bounding box coordinates
                min_x = min(int(lm.x * img_w) for lm in face_landmarks.landmark)
                min_y = min(int(lm.y * img_h) for lm in face_landmarks.landmark)
                max_x = max(int(lm.x * img_w) for lm in face_landmarks.landmark)
                max_y = max(int(lm.y * img_h) for lm in face_landmarks.landmark)

                # Draw the bounding box
                cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360

                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                else:
                    text = "Forward"

                # Append head movement data
                head_movements.append({
                    "frame": len(head_movements),
                    "x_angle": x,
                    "y_angle": y,
                    "direction": text
                })

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                
                cv2.line(image, p1, p2, (255, 0, 0), 2)

                # Add the text on the image
                cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Frame', image)  # Display the frame

        key = cv2.waitKey(1)  # Wait for a key press, 1ms delay
        if key == 27:
            break

    # Release the camera and destroy all windows after the loop has finished
    cap.release()
    cv2.destroyAllWindows()

    # Create a DataFrame from the head movements data
    df = pd.DataFrame(head_movements)

    # Plot the head movement directions over time
    plt.figure(figsize=(14, 6))

    # Plot for head directions
    plt.subplot(1, 2, 1)
    plt.plot(df['frame'], df['direction'], marker='o')
    plt.title('Head Movement Direction Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Direction')
    plt.xticks(rotation=45)

    # Plot for head angles
    plt.subplot(1, 2, 2)
    plt.plot(df['frame'], df['x_angle'], label='X Angle', marker='o')
    plt.plot(df['frame'], df['y_angle'], label='Y Angle', marker='x')
    plt.title('Head Rotation Angles Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function to start the video capture
start_video_capture()
