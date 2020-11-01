import cv2
import os

path = 'C:/Users/mailt/OneDrive/Desktop/opencvproj/dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./trainer/trainer.yml')

face_cascade = cv2.CascadeClassifier('./Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./Cascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('./Cascades/haarcascade_mcs_mouth.xml')
upper_body = cv2.CascadeClassifier('./Cascades/haarcascade_upperbody.xml')
bw_threshold = 80
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
weared_mask_font_color = (0, 255, 0)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = "Thank You for wearing MASK"
not_weared_mask = "Please wear MASK to defeat Corona"
name = ""
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)
    if(len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(img, "No face found...", org, font, font_scale, (255,0,0), thickness, cv2.LINE_AA)
    elif(len(faces) == 0 and len(faces_bw) == 1):
        cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)
        if(len(mouth_rects) == 0):
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            if (confidence < 100):
                imagePaths = [os.path.join(path,f) for f in os.listdir(path)]             
                for imagePath in imagePaths:
                    id1 = int(os.path.split(imagePath)[-1].split(".")[1])
                    if(id==id1):
                        name = (os.path.split(imagePath)[-1].split(".")[0])
                        name = str(name)
                confidence = "  {0}%".format(round(confidence+20))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))        
            cv2.putText(img, name.upper(), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)              
            cv2.putText(img, "Thank You "+str(name.upper())+" for wearing mask", org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        else:
            for (mx, my, mw, mh) in mouth_rects:
                if(y < my < y + h):
                    cv2.putText(img, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness, cv2.LINE_AA)
                    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                    id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                    if (confidence < 100):
                        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]             
                        for imagePath in imagePaths:
                            id1 = int(os.path.split(imagePath)[-1].split(".")[1])
                            if(id==id1):
                                name = (os.path.split(imagePath)[-1].split(".")[0])
                                name = str(name)
                        confidence = "  {0}%".format(round(confidence+30))
                    else:
                        id = "unknown"
                        confidence = "  {0}%".format(round(100 - confidence))        
                cv2.putText(img, name.upper(), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)              
    cv2.imshow('Mask Detection', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
