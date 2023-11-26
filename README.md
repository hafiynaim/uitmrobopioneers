import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open a webcam stream
cap = cv2.VideoCapture(0)

# Initialize variables for state tracking
hand_raised = False
hand_raised_last_frame = False
wave_detected = False
hand_lowered_last_frame = False
wave_complete = False

# Variables to track wrist positions
left_wrist_prev = None
right_wrist_prev = None

# Initialize time for delaying wave detection
wave_detection_start_time = None
wave_detection_delay = 5  # 5 seconds delay

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(rgb_frame)
    if results.pose_landmarks:
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # If wrists are higher than shoulders, consider it as a raised hand
        hand_raised = (left_wrist.y < left_shoulder.y) or (right_wrist.y < right_shoulder.y)

        if hand_raised and not hand_raised_last_frame:
            print("Raise hand")
            hand_raised_last_frame = True
            wave_complete = False
            wave_detection_start_time = time.time()  # Start the delay timer
        elif not hand_raised and hand_raised_last_frame:
            hand_raised_last_frame = False

        # Calculate horizontal movement of wrists if landmarks are present
        if left_wrist.x != 0 and right_wrist.x != 0 and left_wrist_prev is not None and right_wrist_prev is not None:
            left_wrist_movement = abs(left_wrist.x - left_wrist_prev.x)
            right_wrist_movement = abs(right_wrist.x - right_wrist_prev.x)
            
            # Detect wave if hand is raised, landmarks present, movement exceeds threshold,
            # and wave detection delay has passed
            if (left_wrist_movement > 0.02 or right_wrist_movement > 0.02) and hand_raised and not wave_complete and time.time() - wave_detection_start_time >= wave_detection_delay:
                wave_complete = True
                print("Detect wave")
                
            
        
                # Rep data
                #cv2.putText(image, 'Detect Wave', (15,12), 
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                #cv2.putText(image, str(counter), 
                            #(10,60), 
                            #cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        left_wrist_prev = left_wrist
        right_wrist_prev = right_wrist

        # Reset wave detection if hand is lowered
        if not hand_raised and hand_lowered_last_frame:
            wave_detected = False
            hand_lowered_last_frame = False
            wave_complete = False

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    

    # Display the frame with pose landmarks
    cv2.imshow('Pose Estimation', frame)

    # Exit the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
