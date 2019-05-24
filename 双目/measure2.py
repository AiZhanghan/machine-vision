# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:41:16 2019

@author: Administrator
标准：25.06
测得：26.10
"""

import cv2 as cv
import numpy as np
from scipy.optimize import leastsq

def extrace_object(image, a):
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
#    cv.imshow('blurred', blurred)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    gray = 255  - gray
#    cv.imshow('gray', gray)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 50,
                              param1 = 50, param2 = 60, 
                              minRadius = 0, maxRadius = 0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(gray, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(gray, (i[0], i[1]), 2, (255, 0, 0), 2)
#    cv.imshow('circles', gray)
    hough_circle_center = circles[0, a, :2]
    hough_circle_radii = circles[0, a, 2]
    
    roi_radii = hough_circle_radii + 13
    mask = np.zeros(gray.shape)
    cv.circle(mask, (hough_circle_center[0], hough_circle_center[1]), roi_radii, 1, -1)
#    mask[hough_circle_center[1] - roi_radii: hough_circle_center[1] + roi_radii,
#         hough_circle_center[0] - roi_radii: hough_circle_center[0] + roi_radii] = 1
    mask = mask.astype(np.uint8)
#    cv.imshow('mask', mask)
#    print(mask.dtype, gray.dtype)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    masked_image = gray * mask
#    cv.imshow('masked_image', masked_image)
    
    ret, binary = cv.threshold(masked_image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#    cv.imshow('binary image', binary)
    
    edge_output = cv.Canny(binary, 50, 150)
    cv.circle(edge_output, (hough_circle_center[0], hough_circle_center[1]), roi_radii, 0, 2)
#    print(edge_output.shape)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (100, 100))
    edge_output = cv.morphologyEx(edge_output, cv.MORPH_CLOSE, kernel)
    edge_output = cv.Canny(edge_output, 50, 150)
#    cv.imshow('canny edge', edge_output)
    
    index = np.argwhere(edge_output == 255)
    X_d, Y_d= index[:, 0], index[:, 1]
    
    return X_d, Y_d

def fit_plane(X_W, Y_W, Z_W):
    
    def error(p, x, y, z):
        a, b, c = p
        return a * x + b * y + c * z + 1
    
    p0 = [0, 0, 1 / 516]
    Para = leastsq(error, p0, args = (X_W, Y_W, Z_W))
    a, b, c = Para[0]
    print('a = ', a, '\nb = ', b, '\nc = ', c)
    return a, b, c

def fit_ball(X_W, Y_W, Z_W):
    
    def error(p, x, y, z):
        x0, y0, z0, r = p
        return (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r ** 2
    
    p0 = [X_W.mean(), Y_W.mean(), Z_W.mean(), 12.5]
    Para = leastsq(error, p0, args = (X_W, Y_W, Z_W))
    x0, y0, z0, r = Para[0]
    print('x0 = ', x0, '\ny0 = ', y0, '\nz0 = ', z0, '\nr_ball = ', r)
    return x0, y0, z0, r

if __name__ == '__main__':
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
    #cv.imshow('undistort_image_left', undistort_image_left)
    #cv.imshow('image_right', image_right)
    
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

    rgb_rectify_iamge_l = cv.cvtColor(undistort_image_left, cv.COLOR_GRAY2BGR)
    X_d_left, Y_d_left = extrace_object(rgb_rectify_iamge_l, 0)
    
    rgb_rectify_iamge_r = cv.cvtColor(undistort_image_right, cv.COLOR_GRAY2BGR)
    X_d_right, Y_d_right = extrace_object(rgb_rectify_iamge_r, 1)
    #找上下左右四个点进行匹配
    left_up = np.round(np.mean(np.argwhere(X_d_left == min(X_d_left)))).astype(np.int)
    left_dowm = np.round(np.mean(np.argwhere(X_d_left == max(X_d_left)))).astype(np.int)
    left_left = np.round(np.mean(np.argwhere(Y_d_left == min(Y_d_left)))).astype(np.int)
    while left_left not in np.argwhere(Y_d_left == min(Y_d_left)):
        left_left += 1  
    left_right = np.round(np.mean(np.argwhere(Y_d_left == max(Y_d_left)))).astype(np.int)
    while left_right not in np.argwhere(Y_d_left == max(Y_d_left)):
        left_right += 1
        
    left = np.array([[X_d_left[left_up],Y_d_left[left_up]],
                     [X_d_left[left_dowm],Y_d_left[left_dowm]],
                     [X_d_left[left_left],Y_d_left[left_left]],
                     [X_d_left[left_right],Y_d_left[left_right]]])

    right_up = np.round(np.mean(np.argwhere(X_d_right == min(X_d_right)))).astype(np.int)
    right_dowm = np.round(np.mean(np.argwhere(X_d_right == max(X_d_right)))).astype(np.int)
    right_left = np.round(np.mean(np.argwhere(Y_d_right == min(Y_d_right)))).astype(np.int)
    if right_left not in np.argwhere(Y_d_right == min(Y_d_right)):
        right_left += 1
    right_right = np.round(np.mean(np.argwhere(Y_d_right == max(Y_d_right)))).astype(np.int)
    if right_right not in np.argwhere(Y_d_right == max(Y_d_right)):
        right_right += 1
        
    right = np.array([[X_d_right[right_up],Y_d_right[right_up]],
                      [X_d_right[right_dowm],Y_d_right[right_dowm]],
                      [X_d_right[right_left],Y_d_right[right_left]],
                      [X_d_right[right_right],Y_d_right[right_right]]])
    #求视差
    d = left[:, 1] - right[:, 1]
    left = np.append(left, d.reshape(-1, 1), axis = 1)
    left = np.append(left, np.ones((4, 1)), axis = 1)
    left = left.T

    p = np.dot(Q, left)
    
    p = p.T
    for i in range(4):
        p[i] = p[i] / p[i, 3]
    p = p[:, :3].T
 
    a, b, c = fit_plane(p[0], p[1], p[2])

    x0, y0, z0, r_ball = fit_ball(p[0], p[1], p[2])
    
    l = (a * x0 + b * y0 + c * z0 + 1) / np.sqrt(a**2 + b**2 + c**2)
    r = np.sqrt(r_ball**2 - l**2)
    d = 2 * r
    print('d = ', d)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

