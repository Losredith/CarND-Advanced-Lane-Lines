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
        
        #polynomial coefficients for the most recent fit before
        self.measured_fit = []
        
        #polynomial coefficients for the most recent fit after
        self.current_fit = [np.array([False])]  
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
#        #difference in fit coefficients between last and new fits
#        self.diffs = np.array([0,0,0], dtype='float') 
        
        #x values for detected line pixels
        self.allx = []  
        #y values for detected line pixels
        self.ally = []

def region_of_interest(img, vertices):

    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    vertices = np.int32([vertices])
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    masked_image = cv2.polylines(mask, np.int32([vertices]),True,(255,0,0),8)
    final_img = cv2.addWeighted(masked_image, 0.5, img, 0.5, 0.)
    return final_img

def draw_curve(img,IM):

    left = []
    right = []
    line = []
    base = np.zeros_like(img)   
    img_size = (img.shape[1], img.shape[0])
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = leftline.current_fit[0]*ploty**2 + leftline.current_fit[1]*ploty + leftline.current_fit[2]
    right_fitx = rightline.current_fit[0]*ploty**2 + rightline.current_fit[1]*ploty + rightline.current_fit[2]
    
    for y in range(img.shape[0]):
        line.append([left_fitx[y],y])
        line.append([right_fitx[y],y])
        left.append([left_fitx[y],y])
        right.append([right_fitx[y],y])
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    if IM is not None:
        masked_image = cv2.fillPoly(base, np.int32([line]), (255,0,0))
        masked_image = cv2.warpPerspective(masked_image, IM, img_size)
        
    else:
        masked_image = cv2.polylines(base, np.int32([left]),False,(255,0,0),5)
        masked_image = cv2.polylines(base, np.int32([right]),False,(255,0,0),5)
    cv2.putText(masked_image, "Left :"+np.str(np.int(leftline.radius_of_curvature))+"m", (320,200),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),3)
    cv2.putText(masked_image, "Right:"+np.str(np.int(rightline.radius_of_curvature))+"m", (320,260),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),3)
    leftline.line_base_pos = (640 - left_fitx[-1])*3.7/(right_fitx[-1] - left_fitx[-1])
    leftline.line_base_pos = 3.7 - leftline.line_base_pos
    offset = leftline.line_base_pos - 1.85
    if offset < 0:
        cv2.putText(masked_image, np.str(round(-offset,2))+"m left of center", (320,320),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),3)
    else:
        cv2.putText(masked_image, np.str(round(offset,2))+"m right of center", (320,320),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),3)
    final_img = cv2.addWeighted(masked_image, 0.3, img, 0.7, 0.)
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
    thresh_max = 120
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


def corners_unwarp(img, mtx, dist):

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
    warped = cv2.warpPerspective(img, M, img_size)
    
    # Return the resulting image and matrix
    return warped,src,dst,M_Inverse

def line_finding(img):
    
    offset = 200
    histogram = np.sum(img[:int(img.shape[0]/2),offset:img.shape[1]-100], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:])
    
#    plt.plot(histogram)
    leftline.allx = []
    leftline.ally = []
    rightline.allx = []
    rightline.ally = []
    
    for i in range(img.shape[0]):
        histogram = np.sum(img[i:i+10,offset:img.shape[1]-100], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        left = histogram[:midpoint]
        right = histogram[midpoint:]
        if(np.amax(left)>9):
            thresh_left=np.where(left > 9)
            if (thresh_left[0][0] > (leftx_base-100)) and (thresh_left[0][-1] < (leftx_base+100)):
                leftx=np.mean(thresh_left[0]) + offset
                leftline.allx.append(leftx)
                leftline.ally.append(i)
                leftx_base = leftx - offset
        if(np.amax(right)>9):
            thresh_right=np.where(right > 9)
            if (thresh_right[0][0] > (rightx_base-100)) and (thresh_right[0][-1] < (rightx_base+100)):
                rightx=np.mean(thresh_right[0]) + midpoint + offset
                rightline.allx.append(rightx)
                rightline.ally.append(i)
                rightx_base = rightx - midpoint - offset
    if (len(leftline.allx)>0 and len(rightline.allx)>0):
        leftline.measured_fit = np.polyfit(leftline.ally, leftline.allx, 2)
        rightline.measured_fit = np.polyfit(rightline.ally, rightline.allx, 2)    
    else:
        print("error")
    
#    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
#    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

#    plt.imshow(img)
#    plt.plot(left_lanex, left_laney, color='red')
#    plt.plot(right_lanex, right_laney, color='red')
#    plt.xlim(0, 1280)
#    plt.ylim(720, 0)
def cal_radius(fitx,fity):
    y_eval = np.max(fity)   
    fit_cr = np.polyfit(np.asarray(fity) * ym_per_pix, np.asarray(fitx) * xm_per_pix, 2)
    # Calculate the new radii of curvature
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5)/np.absolute(2*fit_cr[0])
    return curverad

def line_verification():

    if leftline.best_fit is None:
        leftline.best_fit = leftline.measured_fit
    if leftline.bestx is None:
        leftline.bestx = np.mean(leftline.allx)
    if leftline.current_fit is None:
        leftline.current_fit = leftline.measured_fit
        
    if rightline.best_fit is None:
        rightline.best_fit = rightline.measured_fit
    if rightline.bestx is None:
        rightline.bestx = np.mean(rightline.allx)    
    if rightline.current_fit is None:
        rightline.current_fit = rightline.measured_fit
        
    leftlinebest = leftline.best_fit[0]*100000+leftline.best_fit[1]*10+leftline.best_fit[2]/100
    leftlinecurrent = leftline.measured_fit[0]*100000+leftline.measured_fit[1]*20+leftline.measured_fit[2]/100
    leftlinediff = np.abs(leftlinebest-leftlinecurrent)
    
    rightlinebest = rightline.best_fit[0]*100000+rightline.best_fit[1]*10+rightline.best_fit[2]/100
    rightlinecurrent = rightline.measured_fit[0]*100000+rightline.measured_fit[1]*10+rightline.measured_fit[2]/100
    
    leftlinediff = np.abs(leftlinebest-leftlinecurrent)
    rightlinediff = np.abs(rightlinebest-rightlinecurrent)
    if len(leftline.allx) > 0 and len(rightline.allx) > 0:
        leftxdiff = np.abs((leftline.bestx - np.mean(leftline.allx)))
        rightxdiff = np.abs((rightline.bestx - np.mean(rightline.allx)))
    else:
        leftxdiff = 0
        rightxdiff = 0
    
    leftlineerror.append(leftlinediff)
    rightlineerror.append(rightlinediff)
    leftxerror.append(leftxdiff)
    rightxerror.append(rightxdiff)

#    print(" ")
#    print(leftlinediff)
#    print(rightlinediff)
#    print(leftxdiff)
#    print(rightxdiff)
    
    if leftlinediff < 50 and leftxdiff < 100:
        leftline.best_fit = (leftline.best_fit*4 + leftline.measured_fit*6)/10.0
        if len(leftline.allx) > 0:
            leftline.bestx = (leftline.bestx +np.mean(leftline.allx))/2.0
        if  leftlinediff < 5 and leftxdiff < 10:
            leftline.current_fit = leftline.measured_fit
            if len(leftline.allx) > 0:
                leftline.radius_of_curvature = cal_radius(leftline.allx,leftline.ally)

    if rightlinediff < 50 and rightxdiff < 100:
        rightline.best_fit = (rightline.best_fit*4 + rightline.measured_fit*6)/10.0
        if len(rightline.allx) > 0:
            rightline.bestx = (rightline.bestx +np.mean(rightline.allx))/2.0
        if rightlinediff < 5 and rightxdiff < 10:
            rightline.current_fit = rightline.measured_fit
            if len(rightline.allx) > 0:
                rightline.radius_of_curvature = cal_radius(rightline.allx,rightline.ally)

    
def process_frame(image):
    cal_result = cv2.undistort(image, mtx, dist, None, mtx)
    warp_result,src,dst,M_Inverse = corners_unwarp(cal_result,mtx,dist)
    can_result = canny(warp_result)
    line_finding(can_result)
    line_verification()
    line = draw_curve(cal_result,M_Inverse)
    return line

def process_video(mtx,dist):
    output = 'output_videos/output.mp4'
    clip2 = VideoFileClip('project_video.mp4')#.subclip(20,45)
    clip = clip2.fl_image(process_frame)
    clip.write_videofile(output, audio=False)
    clip.reader.close()
    clip.audio.reader.close_proc()
#    plt.plot(leftlineerror, color='red')
#    plt.plot(leftxerror,color='green')
#    plt.plot(rightxerror,color='black')
#    plt.plot(rightlineerror, color='blue')
    plt.savefig('output_images/error.png')

    
def process_image(image,mtx,dist):
    cal_example = cv2.undistort(cal_image, mtx, dist, None, mtx)
    cal_result = cv2.undistort(image, mtx, dist, None, mtx)
    warp_result,src,dst,M_Inverse = corners_unwarp(cal_result,mtx,dist)
    can_result = canny(warp_result)
    line_finding(can_result)
    line_verification()
    rect1 = region_of_interest(cal_result,src)
    rect2 = region_of_interest(warp_result,dst)
    line1 = draw_curve(warp_result, None)
    line2 = draw_curve(cal_result, M_Inverse)
    # Plot the result
    f, (ax1, ax2, ax3,ax4,ax5,ax6,ax7,ax8,ax9) = plt.subplots(9, 1, figsize=(120, 40))
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
    
    ax9.imshow(cal_example)
    ax9.set_title('Undistort Example', fontsize=12)
    
    f.savefig('output_images/full_figure.png')
    #plt.subplots_adjust(left=0., right=1., top=2., bottom=1.5)

rightlineerror = []
leftlineerror = []
leftxerror= []
rightxerror = []
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
cal_image = mpimg.imread('camera_cal/calibration2.jpg')
cov_image = mpimg.imread('test_images/test1.jpg')
leftline = Line()
rightline = Line()
mtx,dist = cal_cam(cal_image)
#process_image(cov_image,mtx,dist)
process_video(mtx,dist)







