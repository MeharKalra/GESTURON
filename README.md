# GESTURON
A robust hand gesture recognition model working for both static and dynamic hand gestures to provide contactless system control 

# TITLE 
ROBUST HAND GESTRURE RECOGNITION SYSTEM FOR CONTACTLESS SYSTEM CONTROL
# ABSTRACT 
The purpose of this project is to develop an application which would provide more “hands on” interaction and accessibility to general purpose applications. We make 
use of time-tested Conv-Net based methods in conjunction with other few classic computer vision techniques to identify the interest points. Using these interest points we apply our control logic to determine gestures based on the direction and intensity of change in flow vectors, forming a unique matrix representation. We then compare the real time landmark interest point vectors with these matrix representations providing considerably faster performance at inference time. We further implement a background subtraction method using thresholding techniques to further optimize the gesture recognition pipeline by isolating relevant hand movements in complex environments. Additionally, we introduce a frame rate and time threshold mechanism to ensure the validity and consistency of detected gestures supported by a brief ablation study. Furthermore, these recognized gestures can be mapped to application context-specific actions, allowing a more user-friendly design.
