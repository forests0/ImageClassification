# import necessary packages
import cvlib as cv
import cv2
 
# open webcam
webcam = cv2.VideoCapture(0)
 
if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    
 
# loop through frames
while webcam.isOpened():
 
  # read frame from webcam 
  status, frame = webcam.read()
  if not status:
    print("Could not read frame")
    exit()
 
    # apply face detection
  face, confidence = cv.detect_face(frame)
 
  #print(face)
  #print(confidence)
  rate = 15
  title = 'mosaic'
  while True:
    		# 마우스 드래그로 ROI 선택
    x, y, w, h = cv2.selectROI(title, frame, False)
    if w and h:
      roi = frame[y:y+h, x:x+w]

      roi = cv2.resize(roi, (w//rate, h//rate))
      roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)

      frame[y:y+h, x:x+w] = roi
      cv2.imshow(title, frame)
    else:
      break
    cv2.imshow("Real-time face detection", frame)
 
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows() 