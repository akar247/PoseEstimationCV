# import torch
# from torch.utils.data import DataLoader, Dataset
# import torch.nn as nn
# import torch.optim as optim
import numpy as np
from PIL import Image
import pandas as pd
import os
import cv2 as cv
# import torchvision.transforms.functional as TF
# import torch.nn.functional as F
import matplotlib.pyplot as plt
# import torchvision.models as models
# import torchvision.utils
import time
import turtle
import numpy as np
import mediapipe as mp

print("start")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cam = cv.VideoCapture(0)

cv.namedWindow("test")
all_frames = []
img_counter = 0
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(image)
        # print(results.pose_landmarks)
        goof = results.pose_landmarks
        single = [[-item.x * 150, -item.y * 150] if type(item) != None else [0,0] for item in goof.landmark]
        # print(goof.landmark[0].x)
        all_frames.append(single)
        cv.imshow("test", frame)

        k = cv.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            print(frame)
            break

cam.release()

cv.destroyAllWindows()


# #Hyper Paramaters
# num_classes = 32
# num_epochs = 20
# batch_size = 20
# learning_rate = 0.01
# error = 2

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class Network(nn.Module):
    
#     def __init__(self,num_classes=32):
#         super().__init__()
#         self.model_name='resnet101'
#         self.model=models.resnet101()
#         self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
#     def forward(self, x):
#         x=self.model(x)
#         return x

# model = Network().to(device)

# #Loss and Optimizer functions
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# loss_min = np.inf


# #The path of the model
# weights_path = 'HumanPose_resnet101.pth'

# #The path of the image being plotted on
# image_path = '030424224.jpg'

# #These are the orignal landmarks for that image
# #


# best_network = Network()
# best_network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')), strict=False) 
# best_network.eval()

# image = frame
# display_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# # image = display_image[y:h, w:x]
# image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


# height, width = image.shape
# Xratio = width/224
# Yratio = height/224
# image = TF.resize(Image.fromarray(image), size=(224, 224))
# image = TF.to_tensor(image)
# image = TF.normalize(image, [0.5], [0.5])

# plt.figure()
# plt.imshow(image.squeeze(0))
# plt.show()

# #This gets the landmarks out of the model
# with torch.no_grad():
#     landmarks = best_network(image.unsqueeze(0))

# image_x = landmarks[0][12].item() + (landmarks[0][12].item() - landmarks[0][14].item())/2   
# image_y = landmarks[0][13].item() + (landmarks[0][13].item() - landmarks[0][15].item())/2 

# correct = [[-(landmarks[0][i].item()-image_x)*Xratio,-(landmarks[0][i+1].item()-image_y)*Yratio] for i in range(0,len(landmarks[0]),2)]

screen = turtle.Screen()
screen.setup(700, 700)
screen.tracer(0)

skk = turtle.Turtle()
skk.width(3)
skk.speed('fastest')
skk.hideturtle()

turtle.bgpic("sunset.png")

def draw_stick_figure(coordinates):
    pelvis = [(coordinates[24][0] + coordinates[23][0])/2,(coordinates[24][1] + coordinates[23][1])/2]
    neck = [(coordinates[12][0] + coordinates[11][0])/2,(coordinates[12][1] + coordinates[11][1])/2]
    upper_neck = [neck[0]+0.25*(neck[0]-pelvis[0]),neck[1]+0.25*(neck[1]-pelvis[1])]
    head_size = np.sqrt((coordinates[12][0]-neck[0])**2 + (coordinates[12][1]-neck[1])**2)
    
    #head
    skk.penup()
    skk.goto(neck)
    skk.pendown()
    skk.goto(upper_neck)
    skk.circle(head_size)
    
    #straw hat
    skk.penup()
    skk.goto([upper_neck[0], upper_neck[1]+1.5*head_size])
    skk.color("yellow")
    skk.pendown()
    skk.begin_fill()
    skk.setheading(0)
    skk.forward(2*head_size)
    skk.left(135)
    skk.forward(head_size*3*np.sqrt(2)/4)
    # skk.left(45)
    # skk.forward(2.5*head_size)
    skk.right(45)
    skk.circle(1.25*head_size,180)
    skk.right(45)
    # skk.left(45)
    skk.forward(head_size*3*np.sqrt(2)/4)
    skk.left(135)
    skk.forward(2*head_size)
    skk.end_fill()
    skk.penup()
    skk.color("red")
    skk.forward(2*head_size)
    skk.left(135)
    skk.forward(head_size*3*np.sqrt(2)/4)
    skk.right(45)
    skk.pendown()
    skk.begin_fill()
    skk.forward(0.25*head_size)
    skk.left(90)
    skk.forward(2.5*head_size)
    skk.left(90)
    skk.forward(0.25*head_size)
    skk.left(90)
    skk.forward(2.5*head_size)
    skk.end_fill()
    skk.color("black")
    
    #body and right leg
    skk.penup()
    skk.goto(coordinates[30])
    skk.pendown()
    skk.dot()
    skk.goto(coordinates[28])
    skk.dot()
    skk.pendown()
    skk.goto(coordinates[26])
    skk.dot()
    skk.goto(coordinates[24])
    skk.dot()
    skk.goto(coordinates[23])
    skk.dot()
    skk.goto(pelvis)
    skk.dot()
    skk.goto(neck)
    skk.dot()
    skk.goto(coordinates[12])
    skk.dot()
    skk.goto(coordinates[11])
    skk.dot()
    
    #left leg
    skk.penup()
    skk.goto(coordinates[23])
    skk.pendown()
    skk.dot()
    skk.goto(coordinates[25])
    skk.dot()
    skk.goto(coordinates[27])
    skk.dot()
    skk.goto(coordinates[29])
    skk.dot()
    
    #pelvis
    skk.penup()
    skk.goto(coordinates[24])
    skk.pendown()
    skk.dot()
    skk.goto(coordinates[23])
    skk.dot()

    #right arm
    skk.penup()
    skk.goto(coordinates[12])
    skk.pendown()
    skk.dot()
    skk.goto(coordinates[14])
    skk.dot()
    skk.goto(coordinates[16])
    skk.dot()
    
    #left arm
    skk.penup()
    skk.goto(coordinates[11])
    skk.pendown()
    skk.dot()
    skk.goto(coordinates[13])
    skk.dot()
    skk.goto(coordinates[15])
    skk.dot()
    
    
    

    
    
for drawing in all_frames:
    skk.clear()
    draw_stick_figure(drawing)
    screen.update()
    time.sleep(0.025)
# draw_stick_figure(all_frames[20])
turtle.getscreen().getcanvas().postscript(file = "output.ps")

turtle.done()