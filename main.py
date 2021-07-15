import datetime
import time
import cv2
import face_recognition
import os
import numpy as np

#使用 DroidCam(APP)
video = 'http://192.168.1.104:4747/video'
imgPath = './image'
images = []
classes = []
List = os.listdir(imgPath)


#讀取圖片,添加類別
for name in List:
    img = cv2.imread(os.path.join(imgPath,name))
    images.append(img)
    classes.append(os.path.splitext(name)[0])

#圖像壓縮
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
#計算fps
def calcFps():
    end_time = time.time()
    time_diff = end_time - start_time

    fps = 1 / time_diff
    fps_text = 'fps : {:.2f}'.format(fps)
    return fps_text

#紀錄並儲存辨識人臉(非重複)
def record(name):
    with open('./record/record.csv','r+') as f:
        dataList = f.readlines()
        nameList = []
        today = datetime.datetime.today()
        time = today.strftime('%Y/%m/%d,%H:%M:%S')
        for line in dataList:
            data = line.split(',')
            nameList.append(data[0])
        if name not in nameList:
            text = f'{name},{time}\n'
            f.writelines(f'{name},{time}\n')

encodeListKnown = findEncodings(images)
print('encoding complete')

cap = cv2.VideoCapture(video)


while True:
    ret,frame = cap.read()
    start_time = time.time()
    #使用CUDA/GPU提升效能(model='cnn')
    frameFaceLocs = face_recognition.face_locations(frame,number_of_times_to_upsample=1,model="cnn")
    encodes = face_recognition.face_encodings(frame,frameFaceLocs)

    for encode,faceLoc in zip(encodes,frameFaceLocs):
        matches = face_recognition.compare_faces(encodeListKnown,encode)
        faceDist = face_recognition.face_distance(encodeListKnown,encode)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classes[matchIndex]
            y1,x2,y2,x1 = faceLoc
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2+20),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+15,y2+15),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),2)
            record(name)

    fps_text = calcFps()
    cv2.putText(frame,fps_text,(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('frame',frame)
cap.release()
cv2.destroyAllWindows()

