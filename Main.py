import sys

import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QFileDialog
from PIL import Image
import cv2
import os


def check_credentials(username, password):
    credentials = [
        ('ad', 'ad')
    ]
    return (username, password) in credentials


class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.usernameLabel = QLabel('Username:', self)
        self.usernameLabel.move(10, 10)
        self.usernameEdit = QLineEdit(self)
        self.usernameEdit.move(80, 10)
        self.passwordLabel = QLabel('Passcode:', self)
        self.passwordLabel.move(10, 40)
        self.passwordEdit = QLineEdit(self)
        self.passwordEdit.move(80, 40)
        self.loginButton = QPushButton('Login', self)
        self.loginButton.move(10, 80)
        self.loginButton.clicked.connect(self.handleLogin)
        self.exitButton = QPushButton('Exit',self)
        self.exitButton.move(110,80)
        self.exitButton.clicked.connect(self.exitSys)
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Login')

    def exitSys(self):
        exit(0)


    def handleLogin(self):
        # check the username and password against the database
        username = self.usernameEdit.text()
        password = self.passwordEdit.text()

        if check_credentials(username, password):
            self.close()
            webcamWindow = CaptureWebcamWindow()
            webcamWindow.show()
        else:
            # display an error message
            self.errorLabel = QLabel('Invalid username or password', self)
            self.errorLabel.move(10, 10)


        #exit(0)
# FACE_DATASET
class CaptureWebcamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.userIdLabel = QLabel('User ID:', self)
        self.userIdLabel.move(10, 10)
        self.userIdEdit = QLineEdit(self)
        self.userIdEdit.move(70, 10)
        self.startButton = QPushButton('Start', self)
        self.startButton.move(10, 50)
        self.startButton.setStyleSheet("background-color:rgb(0, 255, 0)")
        self.startButton.clicked.connect(self.startCapture)
        self.stopButton = QPushButton('Stop', self)
        self.stopButton.move(120, 50)
        self.stopButton.setStyleSheet("background-color:rgb(255, 0, 0)")
        self.stopButton.clicked.connect(self.stopCapture)
        self.stopButton.setEnabled(False)
        self.imageLabel = QLabel(self)
        self.imageLabel.setFixedSize(1000, 400)
        self.imageLabel.move(10, 120)

        # add train button
        self.trainButton = QPushButton('Train', self)
        self.trainButton.move(10, 90)
        self.trainButton.setStyleSheet("background-color:rgb(255, 255, 0)")
        self.trainButton.clicked.connect(self.train)
        # tin nhan thong bao khi nhap train button
        self.messageLabel = QLabel(self)
        self.messageLabel.setText("")
        self.messageLabel.move(400, 10)
        self.messageLabel.setFixedSize(300, 100)

        # add face reconition button
        self.faceButton = QPushButton('Recognition',self)
        self.faceButton.move(120,90)
        self.faceButton.setStyleSheet("background-color: rgb(15, 119, 255);")
        self.faceButton.clicked.connect(self.face_recog)

        # # add a name button
        # self.nameLable = QLabel('Name:',self)
        # self.nameLable.move(240,10)
        # self.nameEdit = QLineEdit(self)
        # self.nameEdit.move(290,10)



        self.setGeometry(300, 300, 1000, 600)
        self.setWindowTitle('Capture Webcam')
        self.show()

    def startCapture(self):
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.face_id = self.userIdEdit.text()
        # self.face_name = self.nameEdit.text()

        user_id = self.userIdEdit.text()
        # name = self.nameEdit.text()
        user_id = int(user_id)

        print("\n [INFO] Initializing face capture for user id:{0}", self.face_id)
        self.count = 0
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 640)  # set video width
        self.cam.set(4, 480)  # set video height
        self.minW = 0.1 * self.cam.get(3)
        self.minH = 0.1 * self.cam.get(4)

        # self.nameEdit.setEnabled(False)
        self.userIdEdit.setEnabled(False)


        self.captureTimer = self.startTimer(100)
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)


    def stopCapture(self):
        self.killTimer(self.captureTimer)
        self.cam.release()

        # self.nameEdit.setEnabled(True)
        self.userIdEdit.setEnabled(True)


        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)

        self.faceButton.setEnabled(True)
        self.stopButton.setEnabled(True)


    def train(self):

        # Path for face image database
        path = 'dataset'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')
                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)
            return faceSamples, ids

        self.messageLabel.setText("\n[INFO] Training faces. It will take a few seconds. Wait...")
        faces, ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.write('trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

        # Print the number of faces trained and end program
        self.messageLabel.setText("\n[INFO] {0} faces trained".format(len(np.unique(ids))))


    def timerEvent(self, event):
        ret, img = self.cam.read()
        img = cv2.flip(img, 1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(self.minW), int(self.minH)),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self.count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(self.face_id) + '.' + str(self.count) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('image', img)

            # Display the frame in the image label
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.imageLabel.setPixmap(QPixmap.fromImage(qImg))

    def face_recog(self):

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # iniciate id counter
        id = 0

        # names related to ids: example ==> Marcelo: id=1,  etc
        names = ['None', 'Toan', 'Lan', 'Face1', 'Face2']


        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video widht
        cam.set(4, 480)  # set video height

        # Define min window size to be recognized as a face
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:

            ret, img = cam.read()
            # img = cv2.flip(img, -1) # Flip vertically

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                # Check if confidence is less them 100 ==> "0" is perfect match
                if (confidence < 100):
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))

                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            cv2.imshow('camera', img)

            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break

        self.captureTimer = self.startTimer(100)
        self.faceButton.setEnabled(False)
        self.stopButton.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    loginWindow = LoginWindow()
    loginWindow.show()
    app.exec_()

if __name__ == '__main__':
    main()
    app = QApplication(sys.argv)
    window = CaptureWebcamWindow()
    sys.exit(app.exec_())
