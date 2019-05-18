# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:33:05 2019

@author: Administrator
"""

import cv2 as cv
import numpy as np
    
#blocksize = 0
#uniquenessRatio = 0
#numDisparities = 0
#bm = cv.StereoBM_create(16, 9)
#
#def stereo_match():
#    bm.setBlockSize(2 * blocksize + 5)
#    bm.setROI1(valid_pix_ROI1)
#    bm.setROI2(valid_pix_ROI2)
#    bm.setPreFilterCap(31)
#    bm.setMinDisparity(0)
#    bm.setNumDisparities(numDisparities * 16 + 16)
#    bm.setTextureThreshold(10)
#    bm.setUniquenessRatio(uniquenessRatio)
#    bm.setSpeckleWindowSize(100)
#    bm.setSpeckleRange(32)
#    bm.setDisp12MaxDiff(-1)
#    disp = bm.compute(undistort_image_left, undistort_image_right)
#    disp8 = disp.convertTo(cv.CV_8U, 255 / ((numDisparities * 16 + 16)*16.))
#    xyz = cv.reprojectImageTo3D(disp, xyz, Q, True)
#    xyz = xyz * 16;
#    cv.imshow("disparity", disp8)

#if __name__ == '__main__':
image_left = cv.imread(r'left75.bmp')
image_right = cv.imread(r'right75.bmp')
image_size = (1628, 1236)
#    cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)
cv.imshow('image_left', image_left)
#    cv.imshow('image_right', image_right)


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
#    cv.imshow('undistort_image_left', undistort_image_left)
#    cv.imshow('undistort_image_right', undistort_image_right)

#    rgb_rectify_iamge_l = cv.cvtColor(undistort_image_left, cv.COLOR_GRAY2BGR)
#    rgb_rectify_iamge_R = cv.cvtColor(undistort_image_right, cv.COLOR_GRAY2BGR)
#    cv.imshow('rgb_rectify_iamge_l', rgb_rectify_iamge_l)
#    cv.imshow('rgb_rectify_iamge_R', rgb_rectify_iamge_R)

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

#cv.namedWindow('disparity')
#
#cv.createTrackbar('blockSize', 'disparity', blocksize, 8, stereo_match)
#cv.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50, stereo_match)
#cv.createTrackbar('numDisparities', 'disparity', numDisparities, 16, stereo_match)

blocksize = 0
uniquenessRatio = 0
numDisparities = 0
bm = cv.StereoBM_create(16, 9)

cv.namedWindow("disparity")

cv.createTrackbar('blockSize', 'disparity', blocksize, 8, lambda x: None)
cv.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50, lambda x: None)
cv.createTrackbar('numDisparities', 'disparity', numDisparities, 16, lambda x: None)

blocksize = cv.getTrackbarPos('blockSize', 'disparity')
uniquenessRatio = cv.getTrackbarPos('uniquenessRatio', 'disparity')
numDisparities = cv.getTrackbarPos('numDisparities', 'disparity')
#cv.createTrackbar("num", "depth", 0, 10, lambda x: None)
#cv.createTrackbar("blockSize", "depth", 5, 255, )

#num = cv.getTrackbarPos("num", "depth")
#blockSize = cv.getTrackbarPos("blockSize", "depth")

#if blockSize % 2 == 0:
#    blockSize += 1
#if blockSize < 5:
#    blockSize = 5

bm.setBlockSize(2 * blocksize + 5)
bm.setROI1(valid_pix_ROI1)
bm.setROI2(valid_pix_ROI2)
bm.setPreFilterCap(31)
bm.setMinDisparity(0)
bm.setNumDisparities(numDisparities * 16 + 16)
bm.setTextureThreshold(10)
bm.setUniquenessRatio(uniquenessRatio)
bm.setSpeckleWindowSize(100)
bm.setSpeckleRange(32)
bm.setDisp12MaxDiff(-1)
disp = bm.compute(undistort_image_left, undistort_image_right)
disp8 = cv.normalize(disp, disp, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
threeD = cv.reprojectImageTo3D(disp.astype(np.float32)/16., Q)
#xyz = xyz * 16;
cv.imshow("disparity", disp8)
    
#stereo = cv.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
#
#disparity = stereo.compute(undistort_image_left, undistort_image_right)

#disp = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
#
#threeD = cv.reprojectImageTo3D(disparity.astype(np.float32)/16., Q)
#
#cv.imshow("depth", disp)

cv.waitKey(0)

cv.destroyAllWindows()