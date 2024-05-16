# インポート (適宜pipでインストールする)
import imutils
import numpy as np
import cv2

# VideoCaptureをオープン
cap = cv2.VideoCapture(0)

# モデルを読み込む(顔検出)
prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# モデルを読み込む（感情分類）
emotion_model = 'model.weights.h5'  # ご自身の感情分類モデルのファイル名に変更してください
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# OpenCVが利用するためにKerasモデルをロードする
emotion_net = cv2.dnn.readNetFromModelOptimizer(emotion_model)

# モデルの入力サイズを取得
emotion_width = 64
emotion_height = 64

# カメラ画像を読み込み，顔検出して表示するループ
while True:
    ret, frame = cap.read()

    # カメラ画像を幅400pxにリサイズ
    img = imutils.resize(frame, width=1000)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # 物体検出器にblobを適用する
    net.setInput(blob)
    detections = net.forward()

    # 顔を検出して感情を分析するループ
    for i in range(0, detections.shape[2]):
        # confidenceの値が0.5以上の領域のみを検出結果として描画する
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # 対象領域のバウンディングボックスの座標を計算する
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 顔画像を切り抜いて感情を分析する
            face = img[startY:endY, startX:endX]
            face_blob = cv2.dnn.blobFromImage(cv2.resize(face, (emotion_width, emotion_height)), 1.0, (emotion_width, emotion_height), (0, 0, 0), swapRB=True, crop=False)

            emotion_net.setInput(face_blob)
            emotion_preds = emotion_net.forward()
            emotion_index = np.argmax(emotion_preds)
            emotion_label = emotion_labels[emotion_index]

            # 感情ラベルを描画する
            cv2.putText(img, emotion_label, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Detection", img)
    k = cv2.waitKey(1)&0xff
    if k == ord('s'):
        cv2.imwrite("./output.jpg", img) # ファイル保存
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
