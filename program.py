import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import serial
from time import sleep
import requests

url = "http://monitoring-jentik.xyz/handle_post.php"

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 50)
fontScale = 1
fontColor = (0, 255, 0)
thickness = 2
lineType = 2

ser = serial.Serial("/dev/cu.usbserial-1410", 9600, timeout=0.1)
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=os.path.join(os.path.dirname(__file__), "bobot/weights/best.pt"),
    force_reload=False,
)
vid = cv2.VideoCapture(0)
saved_file = os.path.join(os.path.dirname(__file__), 'temp.jpg')

while True:
    ret, frame = vid.read()
    img_data = frame

    results = model(img_data)
    total_detected = len(results.pandas().xyxy[0])
    img_result = np.squeeze(results.render())

    cv2.putText(
        img_result,
        "Terdeteksi {} jentik".format(total_detected),
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType,
    )
    cv2.imshow("Frame", img_result)

    strnya = ser.readline().decode("utf-8").rstrip()
    if len(strnya) > 0:
        if strnya[0] == "*" and strnya[len(strnya) - 1] == "#":
            print("Valid")
            sanitized = strnya[1:len(strnya)-1].split("|")

            cv2.imwrite(saved_file, img_result )
            print("Lattitude: {}\tLongitude: {}".format(sanitized[1], sanitized[2]))
            cv2.putText(
                img_result,
                "Mengirimkan hasil",
                (10,85),
                font,
                fontScale,
                (0,0,255),
                thickness,
                lineType,
            )
            cv2.imshow("Frame", img_result)
            payload={'lat': sanitized[1],'lng': sanitized[2],'jumlah': total_detected}
            files=[('file',('temp.jpg',open(saved_file,'rb'),'image/jpeg'))]
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15'}
            response = requests.request("POST", url, headers=headers, data=payload, files=files)
            print(response.text)
            sleep(1)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()

# img = os.path.join(os.path.dirname(__file__), '83.jpg')
# saved_file = os.path.join(os.path.dirname(__file__), 'temp.jpg')
# # img_data = cv2.imread(img)[..., ::-1]
# results = model(img_data)
# total_detected = len(results.pandas().xyxy[0])

# print("Terdeteksi", total_detected)
# print(saved_file)
# cv2.imshow('Detected', np.squeeze(results.render())  )
# cv2.imwrite(saved_file, np.squeeze(results.render()) )
# cv2.waitKey(0)
