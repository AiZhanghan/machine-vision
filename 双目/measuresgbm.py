# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:28:18 2019

@author: Administrator
"""

import cv2 as cv
import numpy as np

def update(val = 0):
    stereo.setBlockSize(cv.getTrackbarPos('window_size', 'bar'))
    stereo.setUniquenessRatio(cv.getTrackbarPos('uniquenessRatio', 'bar'))
    stereo.setNumDisparities(cv.getTrackbarPos('num_disp', 'bar'))
#    stereo.setSpeckleWindowSize(cv.getTrackbarPos('speckleWindowSize', 'bar'))
#    stereo.setSpeckleRange(cv.getTrackbarPos('speckleRange', 'bar'))
#    stereo.setDisp12MaxDiff(cv.getTrackbarPos('disp12MaxDiff', 'bar'))
 
    print ('computing disparity...')
    disp = stereo.compute(undistort_image_left, undistort_image_right).astype(np.float32) / 16.0
    
    cv.imshow('disparity', (disp-min_disp)/num_disp)

image_left = cv.imread(r'left75.bmp')
image_right = cv.imread(r'right75.bmp')
image_size = (1628, 1236)

#cv.imshow('image_left', image_left)
#cv.imshow('image_right', image_right)


camera_matrix_left = np.array([[2.749308632632886e+03, 0.186347592447377, 8.354394370551296e+02],
                               [0, 2.749155984942296e+03, 6.178212235977716e+02],
                               [0, 0, 1]])

distortion_coefficients_left = np.array([-0.100820091519514, 
                                          0.431724708548702,
                                          -1.134338925095490e-04,
                                          1.377909782124157e-04,
                                          0])

camera_matrix_right = np.array([[2.748434768653378e+03, -0.318755550218435, 8.382562930597212e+02],
                                [0, 2.748455437300060e+03, 6.286444883548344e+02],
                                [0, 0, 1]])
            
distortion_coefficients_right = np.array([-0.080839682668340, 
                                          0.210038504597218,
                                          -5.195267652317574e-04,
                                          1.796251214555310e-04,
                                          0])

rotation_right = np.array([[0.911440331436569, 0.014345624787515, -0.411182107198566],
                           [-0.008997347788119, 0.999847916434187, 0.014939602906222],
                           [0.411333891095271, -0.009917008201661, 0.911430788916389]])

rotation_right = np.linalg.inv(rotation_right)

translation_right = np.array([-1.851843269088477e+02, -1.884515069525864, 44.167273334501800])
   
R1, R2, P1, P2, Q, valid_pix_ROI1, valid_pix_ROI2 = cv.stereoRectify(camera_matrix_left, 
                                                                     distortion_coefficients_left,
                                                                     camera_matrix_right, 
                                                                     distortion_coefficients_right,
                                                                     image_size, 
                                                                     rotation_right, 
                                                                     translation_right,
                                                                     flags = cv.CALIB_ZERO_DISPARITY,
                                                                     alpha = -1,
                                                                     newImageSize = image_size)

mapx_left, mapy_left = cv.initUndistortRectifyMap(camera_matrix_left, 
                                                  distortion_coefficients_left,
                                                  R1, P1, image_size, 
                                                  m1type = cv.CV_32FC1)

mapx_right, mapy_right = cv.initUndistortRectifyMap(camera_matrix_right, 
                                                    distortion_coefficients_right, 
                                                    R2, P2, image_size,
                                                    m1type = cv.CV_32FC1)

gray_left = cv.cvtColor(image_left, cv.COLOR_BGR2GRAY)
gray_right = cv.cvtColor(image_right, cv.COLOR_BGR2GRAY)
undistort_image_left = cv.remap(gray_left, mapx_left, mapy_left, cv.INTER_LINEAR)
undistort_image_right = cv.remap(gray_right, mapx_right, mapy_right, cv.INTER_LINEAR)

sf = 600 / max(image_size[0], image_size[1])
w = np.round(image_size[0] * sf).astype(np.int)
h = np.round(image_size[1] * sf).astype(np.int)

canvas_left_part = cv.resize(undistort_image_left, (w, h))
vroiL = np.array([np.round(valid_pix_ROI1[0] * sf), np.round(valid_pix_ROI1[1] * sf),
                  np.round(valid_pix_ROI1[2] * sf), np.round(valid_pix_ROI1[3] * sf)]).astype(np.int)
cv.rectangle(canvas_left_part, (vroiL[0], vroiL[1]), 
             (vroiL[0] + vroiL[2], vroiL[1] + vroiL[3]), (0, 0, 255), 3, 8)

canvas_right_part = cv.resize(undistort_image_right, (w, h))
vroiR = np.array([np.round(valid_pix_ROI2[0] * sf), np.round(valid_pix_ROI2[1] * sf),
                  np.round(valid_pix_ROI2[2] * sf), np.round(valid_pix_ROI2[3] * sf)]).astype(np.int)
cv.rectangle(canvas_right_part, (vroiR[0], vroiR[1]), 
             (vroiR[0] + vroiR[2], vroiR[1] + vroiR[3]), (0, 0, 255), 3, 8)

canvas = np.append(canvas_left_part, canvas_right_part, axis = 1)

for i in range(0, len(canvas), 16):
    cv.line(canvas, (0, i), (len(canvas[0]), i), (0, 255, 0), 1, 8)
    
cv.imshow('rectified', canvas)

window_size = 5
min_disp = 16
num_disp = 192-min_disp
blockSize = window_size
uniquenessRatio = 1
#speckleRange = 12
#speckleWindowSize = 3
#disp12MaxDiff = 200
#P1 = 600
#P2 = 2400

cv.namedWindow("disparity", cv.WINDOW_AUTOSIZE)
cv.namedWindow("bar", cv.WINDOW_AUTOSIZE)
#cv.createTrackbar('speckleRange', 'bar', speckleRange, 50, update)    
cv.createTrackbar('window_size', 'bar', window_size, 21, update)
#cv.createTrackbar('speckleWindowSize', 'bar', speckleWindowSize, 200, update)
cv.createTrackbar('uniquenessRatio', 'bar', uniquenessRatio, 50, update)
#cv.createTrackbar('disp12MaxDiff', 'bar', disp12MaxDiff, 250, update)
cv.createTrackbar('num_disp', 'bar', num_disp, 100, update)

stereo = cv.StereoSGBM_create(minDisparity = min_disp,
                               numDisparities = num_disp,
                               blockSize = window_size,
                               uniquenessRatio = uniquenessRatio,
                               )

update()

cv.waitKey(0)

cv.destroyAllWindows()