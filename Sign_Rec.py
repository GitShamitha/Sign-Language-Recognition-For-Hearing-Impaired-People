import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import mediapipe as mp

# Load model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Define actions and colors
actions = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']  # Update with your actions
colors = [(245, 117, 16) for _ in range(len(actions))]  # Update with your colors

# Function to visualize probabilities
def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Function for mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
    return image, results

# Function to extract keypoints
def extract_keypoints(results):
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
    return keypoints

# Function to extract keypoints from image
def extract_keypoints_from_image(image, hands):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
    return keypoints

# Streamlit app
st.title('Sign Language Recognition For Hearing-Impaired People')
st.markdown(
    """This system is developed to help hearing-impaired people learn about the alphabet using the system."""
)

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = None
is_camera_on = False

# Function to process uploaded image
def process_uploaded_image(uploaded_file):
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    keypoints = extract_keypoints_from_image(image, hands)
    if len(keypoints) == 63:  # Check if keypoints were extracted
        # Repeat the keypoints to create a sequence of 30 frames
        keypoints_sequence = [keypoints] * 30
        prediction = model.predict(np.expand_dims(keypoints_sequence, axis=0))[0]
        predicted_action = actions[np.argmax(prediction)]
        st.image(image, caption=f'Predicted Alphabet: {predicted_action}', use_column_width=True)
    else:
        st.write("No hand detected in the image. Please try with another image.")


# Webcam feed function
def webcam_feed():
    global cap,is_camera_on
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.8
    accuracy = []

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened() and is_camera_on:
            ret, frame = cap.read()
            if not ret:
                break

            crop_frame = frame[40:400, 0:300]
            frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
            image, results = mediapipe_detection(crop_frame, hands)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            try:
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))

                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                                    accuracy.append(str(res[np.argmax(res)] * 100))
                            else:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)] * 100))

                    if len(sentence) > 1:
                        sentence = sentence[-1:]
                        accuracy = accuracy[-1:]

                cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
                cv2.putText(frame, "Output: -" + ' '.join(sentence) + ''.join(accuracy), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Viz probabilities
                # frame = prob_viz(res, actions, frame, colors, threshold)

            except Exception as e:
                pass

            cv2.imshow('OpenCV Feed', frame)
            global key, key1
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break  # Exit the loop if 'q' is pressed
            key1 = cv2.waitKey(10) & 0xFF
            if key1 == ord('r'):
    # Restart the camera feed if 'r' is pressed
                cap.release()  # Release the camera
                cv2.destroyAllWindows()  # Close all OpenCV windows
                cap = cv2.VideoCapture(0)  # Reinitialize the camera capture

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

# Sidebar selection
activities = ["Home","Images of Sign with Alphabet", "Upload Image","Webcam Sign Language Detection", "About"]
choice = st.sidebar.selectbox("Select Activity", activities)

if choice == "Home":
   st.write("""
    <h5 style='font-family: Arial, sans-serif; font-weight: bold;'>Welcome to the Sign Language Recognition Application</h5>
    <p style='font-family: Arial, sans-serif;'>Select 'Webcam Sign Language Detection' to start detecting sign language gestures in real-time.</p>
    <p style='font-family: Arial, sans-serif;'>Select 'Upload Image' to upload an image and determine which alphabet it represents.</p>
    """, unsafe_allow_html=True)


elif choice == "Images of Sign with Alphabet":
    st.subheader("Images of Sign with Alphabet")
    st.image("american_sign_language.png", caption='Example Image', use_column_width=True)
    
elif choice == "Upload Image":
    st.subheader("Upload Image of Sign")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        process_uploaded_image(uploaded_file)

elif choice == "Webcam Sign Language Detection":
    st.header("Webcam Live Feed")
    key1 = st.button("Start")  # Add a button
    key = st.button("Stop")    # Add a button to stop the feed

    if key1 and not is_camera_on:  # Check if start button is clicked and camera is not already on
        is_camera_on = True
        webcam_feed()
    elif key and is_camera_on:  # Check if stop button is clicked and camera is on
        is_camera_on = False
        
elif choice == "About":
    st.subheader("About this app")
    st.markdown(
        """
       This application leverages cutting-edge technologies such as OpenCV, TensorFlow, and MediaPipe 
       to provide real-time detection and interpretation of sign language gestures. 
       It aims to bridge communication gaps by enabling seamless interaction between 
       hearing-impaired individuals and the hearing community. The system offers accurate recognition 
       of a wide range of sign gestures, empowering users with effective communication tools. 
       It is designed to be intuitive and user-friendly, making it accessible for both beginners and advanced users 
       interested in learning or practicing sign language. 
        """
    )
