# Coke Detection on Google Glass #
This application enable user to detect coke bottles. It uses OpenCV for Android and JavaCV.

## Setup ##
(copied from *https://github.com/space150/google-glass-playground/tree/master/OpenCVFaceDetection*) 

1. Install and configure the OpenCV4Android SDK. Thorough instructions can be found in the OpenCV4Android SDK tutorial.
2. Google Glass does not have the Google Play store installed, so you will need to manually install the OpenCV Manager apk. Google Glass is running armeabi-v7a, so the OpenCV_x.x.x_Manager_x.x_armv7a-neon.apk manager apk is needed.
3. Update the library reference in this project to point to your OpenCV4Android library.
4. Build and run the project.

## Usage ##
* At normal use application does not detect faces.
* If you tap with three fingers you reach Main Menu. Currently, a bug in the screen resolution selection requires you to select the resolution with a single tap, and choose 512x288, or whatever resolution your Glass is. From there, you select Face Recognition (haven't changed this from the project I cloned from), and choose LBP Classifier.
