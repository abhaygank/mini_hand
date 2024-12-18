from keras.models import model_from_json
import cv2
import numpy as np
import mediapipe as mp

# Load the trained model
json_file = open("18_12_word(3).json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("18_12_word(3).h5")

# Function to preprocess the image (resize it to 200x200 and normalize)
def extract_features(image):
    feature = np.array(image)
    feature = cv2.resize(feature, (200, 200))  # Resize to 200x200
    feature = feature.reshape(1, 200, 200, 1)  # Reshape to the model's expected input shape
    return feature / 255.0  # Normalize the image

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)
label = ['HELLO', 'I LOVE YOU', 'NO', 'WHICH', 'YES']

while True:
    _, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for MediaPipe
    results = hands.process(frame_rgb)  # Process frame for hand tracking

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate bounding box of the hand
            h, w, c = frame.shape
            x_min = int(min(lm.x for lm in hand_landmarks.landmark) * w)
            y_min = int(min(lm.y for lm in hand_landmarks.landmark) * h)
            x_max = int(max(lm.x for lm in hand_landmarks.landmark) * w)
            y_max = int(max(lm.y for lm in hand_landmarks.landmark) * h)

            # Extract the hand region
            x_min, y_min = max(0, x_min - 20), max(0, y_min - 20)  # Add some padding
            x_max, y_max = min(w, x_max + 20), min(h, y_max + 20)
            hand_roi = frame[y_min:y_max, x_min:x_max]

            if hand_roi.size > 0:
                hand_roi_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                hand_roi_processed = extract_features(hand_roi_gray)

                # Predict using the model
                pred = model.predict(hand_roi_processed)
                prediction_label = label[pred.argmax()]

                # Draw the prediction on the frame
                cv2.rectangle(frame, (x_min, y_min - 40), (x_max, y_min), (0, 165, 255), -1)
                accu = "{:.2f}".format(np.max(pred) * 100)
                cv2.putText(frame, f'{prediction_label}  {accu}%', (x_min + 10, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the output frame
    cv2.imshow("Hand Sign Detection", frame)

    # Exit on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
