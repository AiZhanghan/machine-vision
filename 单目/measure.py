# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:56:14 2019

@author: Administrator
棋盘格大小15mm
圆半径20mm
测得：19.91mm
"""

import cv2 as cv
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.patches import Circle
from scipy.optimize import leastsq

def undistort(image, camera_matrix, distortion_coefficients):
    h, w = image.shape[:2]
    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, 
                                                     distortion_coefficients,
                                                     (w,h), 1, (w,h))
    dst = cv.undistort(image, camera_matrix, distortion_coefficients,
                       None, new_camera_mtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imshow('undistort_image', dst)
    cv.imwrite('undistort_image.png', dst)
    return dst, new_camera_mtx

def extrace_object(image):
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
#    cv.imshow('blurred', blurred)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    gray = 255 - gray
    cv.imwrite('gray.png', gray)
    cv.imshow('gray', gray)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 18,
                              param1 = 100, param2 = 40, 
                              minRadius = 0, maxRadius = 0)
    circles = np.uint16(np.around(circles))
#    for i in circles[0, :]:
#        cv.circle(gray, (i[0], i[1]), i[2], (0, 0, 255), 2)
#        cv.circle(gray, (i[0], i[1]), 2, (255, 0, 0), 2)
#    cv.imshow('circles', gray)
    hough_circle_center = circles[0, 0, :2]
    hough_circle_radii = circles[0, 0, 2]
    
    roi_radii = hough_circle_radii + 7
    mask = np.zeros(gray.shape)
    mask[hough_circle_center[1] - roi_radii: hough_circle_center[1] + roi_radii,
         hough_circle_center[0] - roi_radii: hough_circle_center[0] + roi_radii] = 1
    mask = mask.astype(np.uint8)
#    cv.imshow('mask', mask)
#    print(mask.dtype, gray.dtype)
    masked_image = gray * mask
    cv.imshow('masked_image', masked_image)
    cv.imwrite('masked_image.png', masked_image)
    
    ret, binary = cv.threshold(masked_image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('binary image', binary)
    cv.imwrite('binary_image.png', binary)
    
    edge_output = cv.Canny(binary, 50, 150)
#    print(edge_output.shape)
    cv.imshow('canny edge', edge_output)
    cv.imwrite('canny_edge.png', edge_output)
    
    index = np.argwhere(edge_output == 255)
    X_d, Y_d = index[:, 0], index[:, 1]
    
    return X_d, Y_d

def coordinate_system_transformation(X_d, Y_d, new_camera_mtx, 
                                     rotation_matrix, translation_vectors):
    A = np.linalg.inv(np.dot(new_camera_mtx, 
                             np.append(rotation_matrix[:, :-1], 
                                       np.array([translation_vectors]).T, 
                                       axis = 1)))
    
    coordinate_World = np.dot(A, np.append(np.append(X_d.reshape(1, -1), 
                                                     Y_d.reshape(1, -1), axis = 0), 
                                                    np.ones(X_d.shape).reshape(1, -1), axis = 0))
    
    coordinate_World = coordinate_World / coordinate_World[2]
    X_W, Y_W = coordinate_World[0], coordinate_World[1]
    
    return X_W, Y_W

def fit_circle(X_W, Y_W):
    
    def error(p, x, y):
        x0, y0, r = p
        return (x - x0) ** 2 + (y - y0) ** 2 - r ** 2
    
    p0 = [X_W.mean(), Y_W.mean(), 20]
    Para = leastsq(error, p0, args = (X_W, Y_W))
    x0, y0, r = Para[0]
    print('x0 = ', x0, 'y0 = ', y0, 'r = ', r)
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    cir1 = Circle(xy = (x0, y0), radius=r, alpha=0.4)
#    ax.add_patch(cir1)
#    plt.scatter(X_W, Y_W, color="green", label="样本数据", linewidth = 0.2) 
#    plt.axis('scaled')
#    plt.axis('equal')
#    plt.show()
    
if __name__ == '__main__':
    src = cv.imread(r'WIN_20190423_12_49_57_Pro.jpg')
    cv.imshow('input image', src)
    cv.imwrite('input_image.png', src)
    
    camera_matrix = np.array([[9.239681093184738e+02, -0.015369582553920, 6.546624748490717e+02],
                              [0, 9.254082497754611e+02, 4.041001652378953e+02],
                              [0, 0, 1]])
            
    distortion_coefficients = np.array([0.063290013571740, 
                                        -0.202207841709460,
                                        0.001191786106439,
                                        -0.001067797642135,
                                        0])
            
    rotation_matrix = np.array([[0.9999, -0.0027, 0.0100],
                                [0.0023, 0.9991, 0.0434],
                                [-0.0101, -0.0434, 0.9990]])
        
    rotation_matrix = np.linalg.inv(rotation_matrix)
    
    translation_vectors = np.array([-29.299440599654513, -62.119555729701034, 2.631484750578994e+02])
    
    undistort_image, new_camera_mtx = undistort(src, camera_matrix, distortion_coefficients)
    
    X_d, Y_d = extrace_object(undistort_image)
    
    X_W, Y_W = coordinate_system_transformation(X_d, Y_d, new_camera_mtx, 
                                                rotation_matrix, translation_vectors)
    
    fit_circle(X_W, Y_W)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()