import os
import subprocess
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QDateTime, QUrl
import requests

class Ui_MainWindow(object):
    a, b = 0, 0  # UI element offsets
    x, y = 1280, 720  # Window size

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(self.x, self.y)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("""
        * { color:#ffffff; }
        #MainSide { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(75, 0, 130, 255), stop:1 rgba(30, 144, 255, 255)); }
        #LeftSide { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(75, 0, 130, 255), stop:1 rgba(30, 144, 255, 255)); }
        #RightSide { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(75, 0, 130, 255), stop:1 rgba(30, 144, 255, 255)); }
        """)
        self.centralwidget.setObjectName("centralwidget")

        self.MainSide = QtWidgets.QWidget(self.centralwidget)
        self.MainSide.setGeometry(QtCore.QRect(self.a, self.b, self.x, self.y))

        # Right Sidebar
        self.RightSide = QtWidgets.QWidget(self.MainSide)
        self.RightSide.setGeometry(QtCore.QRect(960, 60, self.x // 4, int(self.y / 1.2)))
        self.RightSide.setObjectName("RightSide")

        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(10)

        # Inc Button
        self.Lock = self.create_button(self.RightSide, 20, 20, "     Inc", "plus.png")

        # Dec Button
        self.Bluetooth = self.create_button(self.RightSide, 20, 120, "     Dec", "minus.png")

        # Display number label
        self.number_display = QtWidgets.QLabel(self.RightSide)
        self.number_display.setGeometry(QtCore.QRect(20, 80, 150, 40))  # Place it below the buttons
        self.number_display.setStyleSheet("font-size: 20px; color: gray;")
        self.number_display.setFont(font)
        self.number_display.setAlignment(QtCore.Qt.AlignCenter)
        self.number_value = 0  # Initial value

        # Update the number display
        self.update_number_display()

        # Modify the Inc and Dec buttons to update the number
        self.Lock.clicked.connect(self.increment_number)
        self.Bluetooth.clicked.connect(self.decrement_number)

        # Other Buttons
        self.Navigation = self.create_button(self.RightSide, 20, 220, "     Navigation", "map-marker.png", self.open_map_in_browser)
        self.SignDetection = self.create_button(self.RightSide, 20, 320, "     Bump Detection", "bump.png", self.run_bump_detection)
        self.Radio = self.create_button(self.RightSide, 20, 420, "     Radio", "radio.png", self.open_radio_in_browser)
        self.Home = self.create_button(self.RightSide, 20, 520, "     Home", "home.png", self.return_home)

        # Left Side (Map Display)
        self.LeftSide = QtWidgets.QWidget(self.MainSide)
        self.LeftSide.setGeometry(QtCore.QRect(self.a, self.b + 10, int(self.x / 1.33), self.y))

        # Time Label (Top Right)
        self.Time = QtWidgets.QLabel(self.MainSide)
        self.Time.setGeometry(QtCore.QRect(970, 10, self.x // 13, self.y // 18))
        self.Time.setStyleSheet("font-size: 16px; color: gray;")
        self.Time.setFont(font)

        # Temperature Label (Below Time)
        self.temp = QtWidgets.QLabel(self.MainSide)
        self.temp.setGeometry(QtCore.QRect(1150, 10, 150, 40))
        self.temp.setStyleSheet("font-size: 16px; color: gray;")
        self.temp.setFont(font)
        self.temp.setAlignment(QtCore.Qt.AlignCenter)

        # Set up timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)

        self.temp_timer = QTimer()
        self.temp_timer.timeout.connect(self.update_temperature)
        self.temp_timer.start(60000)

        self.update_time()
        self.update_temperature()

        # Image Label for symbolic.png with gradient background
        self.image_label = QtWidgets.QLabel(self.LeftSide)
        self.image_label.setGeometry(QtCore.QRect(0, 0, 960, 720))  # Set larger size to cover the area
        self.image_label.setStyleSheet("""
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(75, 0, 130, 255), stop:1 rgba(30, 144, 255, 255));
        """)  # Set gradient background
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        # Scale the image to the full size of the label while keeping the aspect ratio
        self.image_label.setPixmap(QtGui.QPixmap("car.png").scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        MainWindow.setCentralWidget(self.centralwidget)

    def create_button(self, parent, x, y, text, icon_path, func=None):
        """ Helper function to create a button with an icon and text """
        button = QtWidgets.QPushButton(parent)
        button.setGeometry(QtCore.QRect(x, y, int(self.x // 4.5), int(self.y // 12)))  # Convert width & height to int
        button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        button.setIcon(QtGui.QIcon(icon_path))
        button.setIconSize(QtCore.QSize(40, 40))
        button.setText(text)
        button.setStyleSheet("""
        QPushButton {
            background-color: rgba(75, 0, 130, 150);  /* Darker Gradient Violet */
            color: white;
            border-radius: 10px;
            border: 2px solid rgba(30, 144, 255, 200);  /* Lighter Blue Border */
            padding: 10px;
        }
        QPushButton:hover {
            background-color: rgba(30, 144, 255, 180);  /* Gradient Blue on Hover */
        }
        QPushButton:pressed {
            background-color: rgba(75, 0, 130, 200);  /* Darker Violet on Click */
        }
        """)
        if func:
            button.clicked.connect(func)
        return button

    def increment_number(self):
        """ Increase the displayed number """
        self.number_value += 1
        self.update_number_display()

    def decrement_number(self):
        """ Decrease the displayed number """
        self.number_value -= 1
        self.update_number_display()

    def update_number_display(self):
        """ Update the number display label """
        self.number_display.setText(f"Value: {self.number_value}")

    def update_time(self):
        """ Update the time label """
        self.Time.setText(QDateTime.currentDateTime().toString("hh:mm"))

    def toggle_lock(self):
        """ Toggle lock state """
        if self.Lock.text().strip() == "Lock":
            self.Lock.setIcon(QtGui.QIcon("lock-open-alt.png"))
            self.Lock.setText("     Unlock")
        else:
            self.Lock.setIcon(QtGui.QIcon("lock.png"))
            self.Lock.setText("     Lock")

    def open_map_in_browser(self):
        """ Open Google Maps in the default system browser """
        subprocess.run(["xdg-open", "https://www.google.com/maps"])

    def open_radio_in_browser(self):
        """ Open the radio station in the default system browser """
        subprocess.run(["xdg-open", "https://tunein.com/radio/"])

    def return_home(self):
        """ Return to the home screen (hide all other content and show symbolic image) """
        self.image_label.show()  # Show the symbolic image again

    def update_temperature(self):
        """ Fetch and update temperature from OpenWeatherMap API """
        API_KEY = "764ace33f1778d77731a5f44453469f2"
        URL = f"http://api.openweathermap.org/data/2.5/weather?q=Cairo&appid={API_KEY}&units=metric"

        try:
            response = requests.get(URL)
            data = response.json()
            if response.status_code == 200:
                self.temp.setText(f"🌡 {data['main']['temp']}°C")
            else:
                self.temp.setText("❌ Error")
        except:
            self.temp.setText("❌ Error")

    def run_bump_detection(self):
        """ Run the bump detection shell script """
        script_path = "/home/suspension/tflite1/run_bump_detection.sh"
        subprocess.run(f"bash {script_path}", shell=True)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    # ✅ Force GPU Acceleration Fix
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --ignore-gpu-blacklist"

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
