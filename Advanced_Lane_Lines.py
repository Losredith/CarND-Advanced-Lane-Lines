# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:03:03 2017

@author: Losredith
"""

"""


1.Camera calibration -done
2.Distortion correction -done
3.Color/gradient threshold -done
4.Perspective transform -done
5.Detect lane lines-done 
6.Determine the lane curvature

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
nx = 9
ny = 6

# Define a class to receive the characteristics of each line detection

class Line():
    def __init__(self):
#        # was the line detected in the last iteration?
#        self.detected = False  
        
#        # x values of the last n fits of the line
#        self.recent_xfitted = [] 
        
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
#        #difference in fit coefficients between last and new fits
#        self.diffs = np.array([0,0,0], dtype='float') 
        
        #x values for detected line pixels
#        self.allx = None  
        #y values for detected line pixels
#        self.ally = None

def region_of_interest(img, vertices):

    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    vertices = np.int32([vertices])
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    masked_image = cv2.polylines(mask, np.int32([vertices]),True,(255,0,0),8)
    final_img = cv2.addWeighted(masked_image, 0.5, img, 0.5, 0.)
    return final_img

def draw_curve(img, left_fit,right_fit,IM):

    left = []
    right = []
    line = []
    base = np.zeros_like(img)   
    img_size = (img.shape[1], img.shape[0])
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_radius,right_radius = cal_radius(left_fit,right_fit)
    #print(ploty)
    for y in range(img.shape[0]):
        line.append([left_fitx[y],y])
        line.append([right_fitx[y],y])
        left.append([left_fitx[y],y])
        right.append([right_fitx[y],y])
#    line = left + right 
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    if IM is not None:
        masked_image = cv2.fillPoly(base, np.int32([line]), (255,0,0))
        masked_image = cv2.warpPerspective(masked_image, IM, img_size)
        
    else:
        masked_image = cv2.polylines(base, np.int32([left]),False,(255,0,0),5)
        masked_image = cv2.polylines(base, np.int32([right]),False,(255,0,0),5)
    cv2.putText(masked_image, "Left :"+np.str(np.int(left_radius)), (640,200),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),3)
    cv2.putText(masked_image, "Right:"+np.str(np.int(right_radius)), (640,260),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),3)
    final_img = cv2.addWeighted(masked_image, 0.5, img, 0.5, 0.)
    #final_img = cv2.addWeighted(masked_image, 0.5, img, 0.5, 0.)
    return final_img

def cal_cam(cal_img):
    imgpoints = []
    objpoints = []
    objp = np.zeros((nx*ny,3),np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    gray = cv2.cvtColor(cal_img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        return mtx,dist
    else:
        print("error")
    
def canny(img):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    #plt.imshow(s_binary)
    return combined_binary

# Define a function that takes an image, number of x and y points, 
# camera matrix and distortion coefficients
def corners_unwarp(img, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
    [[(img_size[0] / 2) - 50, img_size[1] / 2 + 100],
    [((img_size[0] / 5) + 15), img_size[1]],
    [(img_size[0] * 5 / 6) + 15, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
    dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    M_Inverse = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)
    

    # Return the resulting image and matrix
    return warped,src,dst,M_Inverse

def line_finding(img):

    left_lanex = []
    left_laney = []
    right_lanex = []
    right_laney = []
    offset = 200
    #print(img.shape[1])
    histogram = np.sum(img[:int(img.shape[0]/2),offset:img.shape[1]-100], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:])
#    print(leftx_base)
#    print(rightx_base)
#    leftx_base = np.int(histogram.shape[0]*0.1)
#    rightx_base = histogram.shape[0] *0.8 - midpoint
#    print(leftx_base)
#    print(rightx_base)
    
#    plt.plot(histogram)

    for i in range(img.shape[0]):
        histogram = np.sum(img[i:i+10,offset:img.shape[1]-100], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        left = histogram[:midpoint]
        right = histogram[midpoint:]
        if(np.amax(left)>9):
            thresh_left=np.where(left > 9)
            if (thresh_left[0][0] > (leftx_base-100)) and (thresh_left[0][-1] < (leftx_base+100)):
                leftx=np.mean(thresh_left[0])+offset
                left_lanex.append(leftx)
                left_laney.append(i)
                leftx_base = leftx - offset
        if(np.amax(right)>9):
            thresh_right=np.where(right > 9)
            if (thresh_right[0][0] > (rightx_base-100)) and (thresh_right[0][-1] < (rightx_base+100)):
                rightx=np.mean(thresh_right[0])+midpoint+offset
                right_lanex.append(rightx)
                right_laney.append(i)
                rightx_base = rightx - midpoint - offset
    left_fit = np.polyfit(left_laney, left_lanex, 2)
    right_fit = np.polyfit(right_laney, right_lanex, 2)
    
#    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
#    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

#    plt.imshow(img)
#    plt.plot(left_lanex, left_laney, color='red')
#    plt.plot(right_lanex, right_laney, color='red')
#    plt.xlim(0, 1280)
#    plt.ylim(720, 0)

    return left_fit,right_fit,left_lanex,right_lanex

def cal_radius(left_fit,right_fit):
    y_eval = 720
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
#    print(left_curverad, right_curverad)
    return left_curverad, right_curverad

def line_verification(leftfit,rightfit,leftx,rightx):

    if leftline.best_fit is None:
        leftline.best_fit = leftfit
    if leftline.bestx is None:
        leftline.bestx = np.mean(leftx)
    if leftline.current_fit is None:
        leftline.current_fit = leftfit
        
    if rightline.best_fit is None:
        rightline.best_fit = rightfit
    if rightline.bestx is None:
        rightline.bestx = np.mean(rightx)    
    if rightline.current_fit is None:
        rightline.current_fit = rightfit
        
    leftlinediff = np.abs(np.mean(leftline.best_fit-leftfit))
    rightlinediff = np.abs(np.mean(rightline.best_fit-rightfit))
    leftxdiff = np.abs((leftline.bestx - np.mean(leftx)))
    rightxdiff = np.abs((rightline.bestx - np.mean(rightx)))
    print(" ")
    print(np.abs(leftline.best_fit[0]-leftfit[0]))
    print(np.abs(rightline.best_fit[0]-rightfit[0]))
#    print(leftxdiff)
#    print(rightxdiff)
    
    if leftlinediff < 200 and leftxdiff < 100:
        leftline.best_fit = (leftline.best_fit + leftfit)/2.0
        leftline.bestx = (leftline.bestx +np.mean(leftx))/2.0
        if  leftlinediff < 10 and leftxdiff < 10:
            leftline.current_fit = leftfit
    
    if rightlinediff < 200 and rightxdiff < 100:
        rightline.best_fit = (rightline.best_fit + rightfit)/2.0
        rightline.bestx = (rightline.bestx +np.mean(rightx))/2.0
        if rightlinediff < 10 and rightxdiff < 10:
            rightline.current_fit = rightfit
        
    return leftline.current_fit,rightline.current_fit
    
def process_frame(image):
    cal_result = cv2.undistort(image, mtx, dist, None, mtx)
    warp_result,src,dst,M_Inverse = corners_unwarp(cal_result,mtx,dist)
    can_result = canny(warp_result)
    left,right,leftx,rightx = line_finding(can_result)
    left,right = line_verification(left,right,leftx,rightx)
    line = draw_curve(image, left,right,M_Inverse)
    return line

def process_video(mtx,dist):
    output = 'output_videos/output.mp4'
    clip2 = VideoFileClip('project_video.mp4')#.subclip(20,45)
    clip = clip2.fl_image(process_frame)
    clip.write_videofile(output, audio=False)
    clip.reader.close()
    clip.audio.reader.close_proc()
    
def process_image(image,mtx,dist):
    cal_result = cv2.undistort(image, mtx, dist, None, mtx)
    warp_result,src,dst,M_Inverse = corners_unwarp(cal_result,mtx,dist)
    can_result = canny(warp_result)
    left,right,leftx,rightx = line_finding(can_result)
    
    rect1 = region_of_interest(cov_image,src)
    rect2 = region_of_interest(warp_result,dst)
    line1 = draw_curve(warp_result, left,right,None)
    line2 = draw_curve(cov_image, left,right,M_Inverse)
    # Plot the result
    f, (ax1, ax2, ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(8, 1, figsize=(100, 40))
    f.tight_layout()
    
    ax1.imshow(cov_image)
    ax1.set_title('Original Image', fontsize=12)
    
    ax2.imshow(cal_result)
    ax2.set_title('Undistort Result', fontsize=12)
    
    ax3.imshow(rect1)
    ax3.set_title('Undistort Result with src drawn', fontsize=12)
    
    ax4.imshow(warp_result)
    ax4.set_title('Unwarp Result', fontsize=12)
    
    ax5.imshow(rect2)
    ax5.set_title('Unwarp Result with dst drawn', fontsize=12)
    
    ax6.imshow(can_result)
    ax6.set_title('Canny Result', fontsize=12)
    
    ax7.imshow(line1)
    ax7.set_title('Line Unwarp Result', fontsize=12)
    
    ax8.imshow(line2)
    ax8.set_title('Line Result', fontsize=12)
    
    #plt.subplots_adjust(left=0., right=1., top=2., bottom=1.5)
    

cal_image = mpimg.imread('camera_cal/calibration2.jpg')
cov_image = mpimg.imread('test_images/test11.jpg')
leftline = Line()
rightline = Line()
mtx,dist = cal_cam(cal_image)
#process_image(cov_image,mtx,dist)
process_video(mtx,dist)






