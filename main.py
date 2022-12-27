import cv2
import mediapipe as mp
import math

FRAME_DELAY = 100
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_hands = mp.solutions.hands
mp_fingers = mp_hands.HandLandmark


def run():
    cap = cv2.VideoCapture(0)

    hand = mp_hands.Hands(
      max_num_hands = 5,
      model_complexity = 0,
      min_detection_confidence = 0.5,
      min_tracking_confidence = 0.5,
    )

    while cap.isOpened():
        success, image = cap.read()
        if not success:
          print('Ignoring empty camera frame.')
          continue
        image = cv2.flip(image,1) # BGR
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #RGB

        results = hand.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #BGR
        width, height, _ = image.shape
        #print(width, height)

        if results.multi_hand_landmarks :
          for hand_landmarks in results.multi_hand_landmarks :
            #print('------')
            index_finger_tip = hand_landmarks.landmark[
              mp_fingers.INDEX_FINGER_TIP
            ]
            #print(f'{index_finger_tip.x}, {index_finger_tip.y}')
            #print(hand_landmarks.landmark[mp_fingers.INDEX_FINGER_TIP])
            #print('------')
            angle = getAngle(
              hand_landmarks.landmark[mp_fingers.WRIST],
              hand_landmarks.landmark[mp_fingers.INDEX_FINGER_TIP],
              hand_landmarks.landmark[mp_fingers.PINKY_TIP],
              hand_landmarks.landmark[mp_fingers.MIDDLE_FINGER_TIP],
              hand_landmarks.landmark[mp_fingers.RING_FINGER_TIP])
            cv2.putText(
              image,
              text=f'{str(int(index_finger_tip.x * width))}, {str(int(index_finger_tip.y * height))}',
              org=(100,200),
              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=2,
              color=(0,0,0),
              thickness=2
                )
            mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style()
            )
           
        cv2.imshow('MediaPipe Hands', image)
        cv2.waitKey(FRAME_DELAY)
    cap.release()

def getAngle(ps, p1, p2, q1, q2) :
  #print(ps, p1, p2)
  line1 = abs(math.atan((p1.y - ps.y) / (p1.x - ps.x)))
  line2 = abs(math.atan((p2.y - ps.y) / (p2.x - ps.y)))

  qline1 = abs(math.atan((q1.y - ps.y) / (q1.x - ps.x)))
  qline2 = abs(math.atan((q2.y - ps.y) / (q2.x - ps.y)))

  angle = abs(abs(line1- line2) * 180 / math.pi)
  qangle = angle = abs(abs(qline1- qline2) * 180 / math.pi)
  dis1 = getDist(p1,p2)
  dis2 = getDist(q1,q2)
  #print(f'dis1, dis2 : {dis1}, {dis2}')
  #print(f'qang1, qang2, qfangle : {qline1}, {qline2}, {qangle}')
  #print(f'ang1, ang2, fangle : {line1}, {line2}, {angle}')
  if angle > 8 and angle < 65 and qangle > 35 and dis1 < 21.5 and dis1 > 13 :
    print('rock on')
  return angle
  #exit()

def getDist(p1,p2) :
  return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2) * 100
run()