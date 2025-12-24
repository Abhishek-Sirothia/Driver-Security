# Driver_Security


Here is a comprehensive README.md file tailored to your project, Guardian AI.

I have extracted the features, library requirements, and configuration steps directly from your main.py code to ensure it is accurate.

🛡️ Guardian AI: Driver Security System
Guardian AI is a real-time driver monitoring system built with Python and Streamlit. It uses Computer Vision (MediaPipe & OpenCV) to detect signs of drowsiness and distraction, alerting the driver and their emergency contacts immediately via local sound, Telegram, and WhatsApp.

🚀 Key Features
Real-time Drowsiness Detection: Monitors Eye Aspect Ratio (EAR) to detect eye closure.

Distraction Detection: Uses Head Pose Estimation to track if the driver looks away (Left, Right, Down).

Yawn Detection: Monitors Mouth Aspect Ratio (MAR) to detect fatigue.

Sunglasses Mode: A specific mode that disables eye tracking and relies on head pose and mouth movement for drivers wearing shades.

User Calibration: Customizes the sensitivity threshold for each specific driver.

Multi-Channel Alerts:

🔊 Local: Plays an alarm sound (alert.mp3).

📱 Telegram: Sends an alert message with the driver's GPS location.

📞 WhatsApp: Triggers a VOIP call via CallMeBot to wake the driver.

Black Box Logging: Records incidents with timestamps and GPS coordinates to a CSV file.

Admin Dashboard: Visualizes incident locations on a map and views logs (Password protected).

🛠️ Installation & Requirements
1. Prerequisites
Ensure you have Python installed (preferably 3.8+).

2. Install Dependencies
Run the following command to install the required libraries found in main.py:

Bash

pip install streamlit opencv-python mediapipe pygame requests pandas numpy winsdk
Note: winsdk is used for Windows Geolocation services. If running on Linux/Mac, the location function in the code may need adjustment.

3. File Setup
Ensure the following files are in your project directory:

main.py: The application code.

alert.mp3: An audio file for the alarm sound.

users.json: (Created automatically) Stores user profiles.

incident_log.csv: (Created automatically) Stores accident data.

⚙️ Configuration
To make the alerting system work, you need to configure the API keys in main.py.

Telegram Setup
Open main.py.

Locate TELEGRAM_TOKEN (Line 23).

Replace the value with your bot token obtained from BotFather.

WhatsApp Call Setup (CallMeBot)
Add the phone number +34 644 24 84 01 to your contacts.

Send the message: "I allow callmebot to send me messages".

You will receive an API Key. This key is entered in the "New User" tab of the application, not hardcoded.

🖥️ Usage Guide
1. Run the Application
Bash

streamlit run main.py
2. Register a New Pilot
Navigate to 👤 New User in the sidebar.

Enter the Name, Telegram Chat ID, WhatsApp Number, and CallMeBot API Key.

Click Save User.

3. Start Monitoring
Go to 🚀 Live Monitor.

Select the Pilot from the dropdown.

(Optional) Click 🔬 Calibrate User to set a custom baseline for your eyes.

Click ACTIVATE DMS to start the camera.

4. View Logs
Go to 📊 Flight Data.

Enter the Admin Password (Default: admin).

View the map of incidents and the raw data logs.

📂 Project Structure
├── main.py              # Core application logic
├── incident_log.csv     # Database of detected incidents
├── users.json           # User configuration database
├── alert.mp3            # (Required) Sound file for alerts
└── README.md            # Project documentation
⚠️ Disclaimer
This software is intended for educational and safety assistance purposes. It should not replace responsible driving habits. The developers are not liable for accidents that occur while using this system.
