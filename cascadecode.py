import cv2
import numpy as np





def duygutespiti(s,path):
    
    img = cv2.imread(path)
    
    face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    
    face_rects = face_detect.detectMultiScale(img)
        
        
        
    (face_x, face_y, w, h) = tuple(face_rects[0])
            
        
            
            
        
    track_window = (face_x, face_y, w, h)
    roi = img[face_y:face_y + h, face_x : face_x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
    cv2.normalize(roi_hist , roi_hist ,0 ,255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)
        
                
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180],1)
        
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    x,y,w,h = track_window
                
    img2 = cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255),5)
    img2 = cv2.putText(img2,s,(int((x+w)+3),int(y+h-10)),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0))
    cv2.imshow("resim", img2)
            

        

