# PFC-ELO
Robot control via deep learning based gesture recognition, Wi-Fi communication and microcontrollers

# Summary

This work consists of two main phases in order to control robots. Initially, as the first phase of the project, a simpler approach was taken, in a way that the robot was controlled via smartphone, using accelerometer data extracted from it. Once this mark was achieved, the need for incrementing this work in order to make the control more robust and easier in the final product became clear. With that purpose, a new phase in the project began where the focus was in controlling the robot via gesture recognition. To accomplish the goals and challenges brought by this new phase, a great number of convolutional neural networks were trained, using Deep Learning and in accordance to the relevant metrics such as accuracy, F1-score and inference time, the best model was selected. In that way, the control system is composed by a user that controls the robot by executing gestures in front of a computer webcam so that these are processed by this Artificial Intelligence, the command associated to the gesture is then sent via internet to a Raspberry Pi 3 located in the robot which in turn receives those commands and sends them to an arduino also located in the robot, so that it controls the motors accordingly. Furthermore, a webcam is placed on the robot and connected to the Raspberry in order to send back visual feedback to the user so the robot can be controlled even if it is in a different location than the controller.
In a broader point of view, the work has the general goal of provide a solution to the demand of activities that present high risk to humans while also presenting an alternative to activities that require human presence in order to be accomplished. Exemples of these activities could be firefighters rescues in buildings or highly damaged structure locations presenting a high risk to people entrance and also the possibility of execution of medical procedures or operations done from distance, without the physician responsible necessarily being physically present.

Keywords: Artificial Intelligence. Deep Learning. Gesture recognition. ConvolutionalNeural Networks. Wi-Fi Communication. Raspberry pi 3. Microcontrollers. DC Motors.

# Datasets

https://drive.google.com/drive/u/0/folders/1Ahqt4-TOgJdnOm3Isbpjm8KMwtdrAt9S

