import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)

# Formality that we have to define before start the detection or the tracking
mpHands = mp.solutions.hands
# Create an object called hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while (True):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # We called the method process under the object hands
    # process method used to process the image frame and gives us the result
    results = hands.process(imgRGB)

    # extract the hands given in the "results"
    # First we need to check if we have captured a hands or not
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Getting the ID of each hand
            # and then getting the position (coordinates) information
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c= img.shape
                # Multiplying the (height, width) of each landmark i
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 12:
                    cv2.circle(img, (cx, cy), 15, (255, 20, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 2/(cTime-pTime)
    pTime = cTime


    cv2.putText(img, str(int(fps)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()

