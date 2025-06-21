import threading
import numpy as np
import cv2
import mediapipe as mp
import pyautogui
import pyttsx3
import speech_recognition as sr
import random
from pynput.mouse import Button, Controller
mouse = Controller()


stop_event = threading.Event()

# numpy import
def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle


def get_distance(landmark_ist):
    if len(landmark_ist) < 2:
        return
    (x1, y1), (x2, y2) = landmark_ist[0], landmark_ist[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])

# Hand Gesture Code
screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)


def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None, None


def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y / 2 * screen_height)
        pyautogui.moveTo(x, y)


def is_left_click(landmark_list, thumb_index_dist):
    return (
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            thumb_index_dist > 50
    )


def is_right_click(landmark_list, thumb_index_dist):
    return (
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90  and
            thumb_index_dist > 50
    )


def is_double_click(landmark_list, thumb_index_dist):
    return (
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist > 50
    )


def is_screenshot(landmark_list, thumb_index_dist):
    return (
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist < 50
    )


def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:

        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = get_distance([landmark_list[4], landmark_list[5]])

        if get_distance([landmark_list[4], landmark_list[5]]) < 50  and get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            move_mouse(index_finger_tip)
        elif is_left_click(landmark_list,  thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif is_screenshot(landmark_list,thumb_index_dist ):
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


def hand_main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    


# Eye Tracking Code
def eye_tracking():
    cam = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    screen_width, screen_height = pyautogui.size()

    while not stop_event.is_set():
        ret, frame = cam.read()
        if not ret:
          print("Error: Camera not found or disconnected!")
          break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape

        if landmark_points:
            landmarks = landmark_points[0].landmark

           
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                if id == 1:
                    screen_x = screen_width / frame_w * x
                    screen_y = screen_height / frame_h * y
                    pyautogui.moveTo(screen_x, screen_y)

            left = [landmarks[145], landmarks[159]]
            if left[0].y - left[1].y < 0.004:
                pyautogui.click()
                pyautogui.sleep(1)

    
        cv2.imshow("Eye Tracking", frame)

        # Check for 'q' key press to stop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            stop_event.set()
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Eye tracking stopped.")


def voice_recognition():
    engine = pyttsx3.init()
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    def speak(text):
        engine.say(text)
        engine.runAndWait()

    speak("Voice control activated.")
    while not stop_event.is_set():
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening for voice commands...")
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio).lower()
                print(f"Command: {command}")

                if "stop" in command:
                    speak("Stopping voice control.")
                    stop_event.set() 
                    break
                elif "move" in command:
                    if "left" in command:
                        pyautogui.moveRel(-100, 0)
                        speak("Moving left")
                    elif "right" in command:
                        pyautogui.moveRel(100, 0)
                        speak("Moving right")
                    elif "up" in command:
                        pyautogui.moveRel(0, -100)
                        speak("Moving up")
                    elif "down" in command:
                        pyautogui.moveRel(0, 100)
                        speak("Moving down")
                elif "click" in command:
                    pyautogui.click()
                    speak("Clicked")
                elif "double click" in command:
                    pyautogui.doubleClick()
                    speak("Double clicked")
                elif "scroll" in command:
                    if "up" in command:
                        pyautogui.scroll(300)
                        speak("Scrolling up")
                    elif "down" in command:
                        pyautogui.scroll(-300)
                        speak("Scrolling down")
        except sr.UnknownValueError:
            print("Could not understand the audio.")
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
            stop_event.set()
            break
    print("Voice recognition stopped.")


if __name__ == "__main__":
   
    eye_tracking_thread = threading.Thread(target=eye_tracking)
    voice_recognition_thread = threading.Thread(target=voice_recognition)
    hand_gesture_thread = threading.Thread(target=hand_main)

    
    eye_tracking_thread.start()
    voice_recognition_thread.start()
    hand_gesture_thread.start()

    
    eye_tracking_thread.join()
    voice_recognition_thread.join()
    hand_gesture_thread.join()

    print("Both threads stopped. Exiting program.")
