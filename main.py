import cv2
import mediapipe as mp
import pyautogui
import time


# ========================================
# 1. MediaPipe Setup
# ========================================

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(options)


# ========================================
# 2. Camera
# ========================================

camera = cv2.VideoCapture(0)

last_action_time = 0
COOLDOWN = 2


# ========================================
# 3. Check Finger
# ========================================

def finger_up(hand, tip, pip):

    return hand[tip].y < hand[pip].y


# ========================================
# 4. Detect Gesture
# ========================================

def detect_gesture(hand):

    index = finger_up(hand, 8, 6)
    middle = finger_up(hand, 12, 10)
    ring = finger_up(hand, 16, 14)
    pinky = finger_up(hand, 20, 18)

    # -----------------------------
    # ☝️ Index finger only
    # -----------------------------

    if index and not middle and not ring and not pinky:
        return "NEXT"


    # -----------------------------
    # ✌️ Index + Middle
    # -----------------------------

    elif index and middle and not ring and not pinky:
        return "PREVIOUS"


    # -----------------------------
    # 🖐 Open Palm
    # -----------------------------

    elif index and middle and ring and pinky:
        return "START"


    # -----------------------------
    # ✊ Fist
    # -----------------------------

    elif not index and not middle and not ring and not pinky:
        return "EXIT"


    return "UNKNOWN"


# ========================================
# 5. Main Loop
# ========================================

while True:

    success, frame = camera.read()

    if not success:
        print("Camera error!")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # ====================================
    # Convert Image
    # ====================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # ====================================
    # Detect Hand
    # ====================================

    result = detector.detect(image)


    gesture = "NO HAND"


    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        gesture = detect_gesture(hand)


        # =================================
        # Draw Hand Points
        # =================================

        for point in hand:

            x = int(point.x * width)
            y = int(point.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        # =================================
        # Control PowerPoint
        # =================================

        current_time = time.time()


        if current_time - last_action_time > COOLDOWN:


            # ☝️ NEXT
            if gesture == "NEXT":

                pyautogui.press("right")

                print("➡️ NEXT SLIDE")

                last_action_time = current_time


            # ✌️ PREVIOUS
            elif gesture == "PREVIOUS":

                pyautogui.press("left")

                print("⬅️ PREVIOUS SLIDE")

                last_action_time = current_time


            # 🖐 START
            elif gesture == "START":

                pyautogui.press("f5")

                print("▶️ START PRESENTATION")

                last_action_time = current_time


            # ✊ EXIT
            elif gesture == "EXIT":

                pyautogui.press("esc")

                print("⏹️ EXIT PRESENTATION")

                last_action_time = current_time
                # ====================================
    # Display Gesture
    # ====================================

    cv2.rectangle(
        frame,
        (20, 20),
        (400, 90),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "Gesture: " + gesture,
        (35, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ====================================
    # Project Title
    # ====================================

    cv2.putText(
        frame,
        "SMART SLIDE CONTROL",
        (20, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ====================================
    # Show Camera
    # ====================================

    cv2.imshow(
        "Smart Slide",
        frame
    )


    # ESC closes Python program
    if cv2.waitKey(1) & 0xFF == 27:
        break


# ========================================
# 6. Cleanup
# ========================================

camera.release()

detector.close()

cv2.destroyAllWindows()