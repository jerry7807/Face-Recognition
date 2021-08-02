# Face-Recognition demo

## 環境配置 
* os Windows
* python 3.8.10
* CUDA 11.4

### API

 * face recognition 1.3.0  
 * opencv-python 4.5.2.54
 * numpy 1.21.0
 * dlib 19.22.0


* GPU :  NVIDIA GeForce GTX 1060

notice: 安裝face recognition時會自動安裝dilb，建議先行安裝避免失敗(需先安裝cmake、boost)   

# 臉部識別
* 比對偵測到之圖片和已知圖片(image資料夾中)是否相似
```python
matches = face_recognition.compare_faces(encodeListKnown,encode)
```
* 以歐式距離最小者來判斷圖片為image資料夾中的誰
```python
faceDist = face_recognition.face_distance(encodeListKnown,encode)
matchIndex = np.argmin(faceDist)
```
