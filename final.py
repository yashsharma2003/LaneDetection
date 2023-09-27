#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[9]:


def cannyi(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    edges=cv2.Canny(blur,100,200)
    return edges


# In[10]:


def region_of_interest(image):
    h=image.shape[0]
    w=image.shape[1]
    triangle=np.array([[(0,h),((w//2),h//2),(w,h)]])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,triangle,255)
    masked_image=cv2.bitwise_and(mask,image)
    return masked_image


# In[11]:


import math 
def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            if(abs(math.atan((y2-y1)/(x2-x1)))>(0.09)):
                 cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),7 )
    return line_image


# In[12]:


cap=cv2.VideoCapture('test3.mp4')

width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer= cv2.VideoWriter('res1.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))


# In[13]:


while(cap.isOpened()):
    ret,frame=cap.read()
    if(ret==True):
        canny=cannyi(frame)
        cropped_img=region_of_interest(canny)
        lines=cv2.HoughLinesP(cropped_img,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
        line_image=display_lines(frame,lines)
        combo=cv2.addWeighted(frame,0.8,line_image,1,1) 
        cv2.namedWindow("Resize", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Resize", 1200, 800)

#         cv2.resizeWindow("Resized_Window", 300, 700)
        writer.write(combo)
        cv2.imshow("Resize",combo)
        if(cv2.waitKey(1)==ord('q')):
            break
    else: 
        break
cap.release()
writer.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




