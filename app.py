from typing import final
from flask import Flask, render_template, request,Response
import cv2
import sys
from mtcnn.mtcnn import MTCNN
import numpy as np
from numpy.core.fromnumeric import resize 
from tensorflow.compat.v1.logging import set_verbosity
from tensorflow.compat.v1.logging import ERROR
from tensorflow.keras.models import load_model
import time

label_expression=['angry','disgusted', 'scared', 'happy', 'sad', 'surprised', 'neutral']
model =load_model('D:\emotion_recognition\model.h5')
detector = MTCNN()

def prediction(img):

    set_verbosity(ERROR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = np.array(img).reshape((1, 48, 48, 1))

    prediction = model.predict(img)
    prediction_max = label_expression[np.argmax(prediction)]
    print(label_expression[np.argmax(prediction)])
	
    return prediction_max

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

font = cv2.FONT_HERSHEY_SIMPLEX 

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/video', methods = ['GET', 'POST'])
def video():
   if request.method == 'POST':
      f = request.files['select_video']
      f.save(r"D:\emotion_recognition\static\file.mp4")
      return render_template('video.html')

@app.route('/imagepred', methods=['GET', 'POST'])
def image():
	image = request.files['select_image']
	image.save(r"D:\emotion_recognition\static\file.jpg")
	image = cv2.imread(r"D:\emotion_recognition\static\file.jpg")
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = detector.detect_faces(image) 
	#to draw faces on image
	for result in faces:
		x, y, w, h = result['box']
		cropped = image[y:y+h, x:x+w]
		resized = cv2.resize(cropped, (48, 48))
		cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 3)
		predict = prediction(resized)
		cv2.putText(image, predict, (x , y -6), font, 1, (0,255,0), 3, cv2.LINE_AA)

	cv2.imwrite(r"D:\\emotion_recognition\\static\\after.jpg", image)
	img = cv2.imread(r"D:\\emotion_recognition\\static\\file.jpg", 0)

	img = cv2.resize(img, (48,48))
	img = img/255
	img = img.reshape(1,48,48,1)
	pred = model.predict(img)
	
	pred = np.argmax(pred)
	if len(faces) == 0 :
		final_pred = "This person looks "+label_expression[pred]
	else :
		final_pred = ""

	return render_template('image.html', data=final_pred)

def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(r"D:\emotion_recognition\static\file.mp4")
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            faces = detector.detect_faces(img)# result
            for result in faces:
                x, y, w, h = result['box']
            cropped = img[y:y+h, x:x+w]
    #        cropped = RGB_frame[y:y+h, x:x+w]
            resized = cv2.resize(cropped, (48, 48))
            #resized = cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_AREA)
            predict = prediction(resized)
            #print(detail)
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 0, 0)
            if predict in ['Happiness','Neutral']:
                color = (0, 255, 0)  # Green
            elif predict in ['Surprise', 'Fear', 'Disgust', 'Sadness']:
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 0, 255)  # Red
            cv2.putText(img, predict, (x + w, y), font, 1, color, 2, cv2.LINE_AA)
            #cv2.putText(frame, str(detail[predict]), (x + w, y+25), font, 1, color, 2, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else: 
            break
        

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
	app.run(debug=True)