import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import vlc
#import time


transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

####
media_player = vlc.MediaPlayer()
media = vlc.Media(r"C:\Users\Mehar Kalra\Desktop\New folder\Project1\kajari.mp4")
media_player.set_media(media)
####

PATH = 'my_own_three_class_net.pth'

net = Net()
net.load_state_dict(torch.load(PATH))
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        new_frame = frame[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        #cv2.imshow('mask', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        #cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow('ori', thresh)
        new_frame1 = new_frame[:,:,0]
        new_frame2 = new_frame[:,:,1]
        new_frame3 = new_frame[:,:,2]
                               
        img_1 = cv2.bitwise_and(new_frame1,new_frame1,mask = thresh)
        img_2 = cv2.bitwise_and(new_frame2,new_frame2,mask = thresh)
        img_3 = cv2.bitwise_and(new_frame3,new_frame3,mask = thresh)
 
        img3 = np.sum(gray,axis = 1)
        indices = [ind for ind in range(len(img3)) if img3[ind]==0]
        if len(indices) < img_1.shape[0]:
            img1_new = np.delete(img_1,indices,0)
            img2_new = np.delete(img_2,indices,0)
            img3_new = np.delete(img_3,indices,0)

        img4 = np.sum(gray,axis = 0)
        indices1 = [ind for ind in range(len(img4)) if img4[ind]==0]
        if len(indices1) < img_1.shape[1]:
            img1_new = np.delete(img1_new,indices1,1)
            img2_new = np.delete(img2_new,indices1,1)
            img3_new = np.delete(img3_new,indices1,1)
            r,c = img1_new.shape
            new_image = np.uint8(np.zeros([r,c,3]))
            new_image[:,:,0] = img1_new[:,:]
            new_image[:,:,1] = img2_new[:,:]
            new_image[:,:,2] = img3_new[:,:]    
            cv2.imshow('to_be_saved',new_image)
            new_image1 = cv2.resize(new_image,(28,28))
            image_tensor = transform(new_image1).float()
            image_tensor = image_tensor.unsqueeze_(0)
            input1 = Variable(image_tensor)
            outputs = net(input1)
            _, predicted = torch.max(outputs.data, 1)
            pred = predicted.item()
            value = media_player.is_playing()
            print(pred)
            if pred==1:
                print('v',value)
                media_player.play()
            else:
                print('no')
                media_player.set_pause(1)
            #else:
                #continue
            #print(value)
            
    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        media_player.set_pause(1)
        
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')