# import pytesseract
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
# img = cv2.imread('img/2.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# img = cv2.blur(img, (3,3))
# # img = cv2.medianBlur(img,5)


# sharp_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# sharpen_img = cv2.filter2D(img, ddepth=-1, kernel=sharp_filter)
# in_range = cv2.inRange(sharpen_img, 0, 30 )
# canny = cv2.Canny(in_range, 10, 100)
# res = in_range + canny
# res = cv2.bitwise_not(res)
# cv2.imshow("res", res)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt


# img = cv.imread('/home/molokeev/Desktop/banki/img/0.jpg', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# blur = cv.medianBlur(img,5)
# cimg = cv.cvtColor(blur,cv.COLOR_GRAY2BGR)
 
# circles = cv.HoughCircles(blur,cv.HOUGH_GRADIENT,1,60,
# param1=200,param2=200,minRadius=0,maxRadius=0)

# circles = np.uint16(np.around(circles))

# height,width = blur.shape
# mask = np.zeros((height,width), np.uint8)

# for i in circles[0,:]:
#     i[2] -= 50
#     cv.circle(mask,(i[0],i[1]),i[2],(255,255,255),-1)
#     break

# ret, mask = cv.threshold(mask, 10, 255, cv.THRESH_BINARY)
# mask_1 = cv.bitwise_not(mask)
# # zmask = np.zeros((height,width), np.uint8)
# img1_bg = cv.bitwise_and(img,img,mask = mask)
# img1_bg = cv.bitwise_or(img1_bg,mask_1)
# cv.imshow('img1_bg', img1_bg)
 
# img1_bg = cv.blur(img1_bg, (3,3))

# sharp_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# sharpen_img = cv.filter2D(img1_bg, ddepth=-1, kernel=sharp_filter)
# in_range = cv.inRange(sharpen_img, 0, 50 )
# canny = cv.Canny(in_range, 10, 30)
# res = in_range + canny
# res = cv.bitwise_not(res)   
# cv.imshow("res", res)

# siftobject = cv.SIFT_create()
# bf = cv.BFMatcher()

# keypoint_main, descriptor_main = siftobject.detectAndCompute(res, None)
# keypointimage_main = cv.drawKeypoints(res, keypoint_main, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imshow('SIFT main', keypointimage_main)


# readimage = cv.imread('template.png')
# grayimage = cv.cvtColor(readimage, cv.COLOR_BGR2GRAY)

# keypoint, descriptor = siftobject.detectAndCompute(grayimage, None)
# keypointimage = cv.drawKeypoints(readimage, keypoint, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imshow('SIFT', keypointimage)

# matches = bf.knnMatch(descriptor_main,descriptor,k=2)

# good = []
# for m,n in matches:
#     if m.distance < 0.5*n.distance:
#         good.append([m])


# print(good)
# img3 = cv.drawMatchesKnn(res,keypoint_main,grayimage,keypoint,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# obj = []
# obj.append(keypoint[good[i].queryIdx].pt)

# scene = []
# scene.append(keypoint_main[good[i].trainIdx].pt)
# hah = cv.findHomography( obj, scene, cv.RANSAC )

# perspectiveTransform( obj_corners, scene_corners, H);

# cv.imshow("img3", img3)

# cv.waitKey(0)
# cv.destroyAllWindows()


from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
 
 
img_object = cv.imread('template.png', cv.IMREAD_GRAYSCALE)
img_scene = cv.imread('img/0.jpg', cv.IMREAD_GRAYSCALE)

if img_object is None or img_scene is None:
    print('Не найдены фотографии!')
    exit(0)

blur = cv.medianBlur(img_scene,5)
cimg = cv.cvtColor(blur,cv.COLOR_GRAY2BGR)
 
circles = cv.HoughCircles(blur,cv.HOUGH_GRADIENT,1,60,
param1=200,param2=200,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))

height,width = blur.shape
mask = np.zeros((height,width), np.uint8)

for i in circles[0,:]:
    i[2] -= 50
    cv.circle(mask,(i[0],i[1]),i[2],(255,255,255),-1)
    break

ret, mask = cv.threshold(mask, 10, 255, cv.THRESH_BINARY)
mask_1 = cv.bitwise_not(mask)
# zmask = np.zeros((height,width), np.uint8)
img1_bg = cv.bitwise_and(img_scene,img_scene,mask = mask)
img1_bg = cv.bitwise_or(img1_bg,mask_1)
cv.imshow('img1_bg', img1_bg)
cv.imshow("img_object_1", img_object)
 
img1_bg = cv.blur(img1_bg, (3,3))

sharp_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpen_img = cv.filter2D(img1_bg, ddepth=-1, kernel=sharp_filter)
in_range = cv.inRange(sharpen_img, 0, 50 )
canny = cv.Canny(in_range, 10, 30)
res = in_range + canny
res = cv.bitwise_not(res)   
# cv.imshow("res", res)



detector = cv.SIFT_create()
keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
keypoints_scene, descriptors_scene = detector.detectAndCompute(res, None)

matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
 
ratio_thresh = 0.75
good_matches = []
for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
 
# img_matches = np.empty((max(img_object.shape[0], res.shape[0]), img_object.shape[1]+res.shape[1], 3), dtype=np.uint8)
# cv.drawMatches(img_object, keypoints_obj, res, keypoints_scene, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
obj = np.empty((len(good_matches),2), dtype=np.float32)
scene = np.empty((len(good_matches),2), dtype=np.float32)
for i in range(len(good_matches)):
    obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
    obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
    scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
    scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
 
H, _ = cv.findHomography(obj, scene, cv.RANSAC)
 
obj_corners = np.empty((4,1,2), dtype=np.float32)
obj_corners[0,0,0] = 0
obj_corners[0,0,1] = 0
obj_corners[1,0,0] = img_object.shape[1]
obj_corners[1,0,1] = 0
obj_corners[2,0,0] = img_object.shape[1]
obj_corners[2,0,1] = img_object.shape[0]
obj_corners[3,0,0] = 0
obj_corners[3,0,1] = img_object.shape[0]
 
scene_corners = cv.perspectiveTransform(obj_corners, H)
 
cv.line(res, (int(scene_corners[0,0,0]), int(scene_corners[0,0,1])),(int(scene_corners[1,0,0]), int(scene_corners[1,0,1])), (0,255,0), 4)
cv.line(res, (int(scene_corners[1,0,0]), int(scene_corners[1,0,1])),(int(scene_corners[2,0,0]), int(scene_corners[2,0,1])), (0,255,0), 4)
cv.line(res, (int(scene_corners[2,0,0]), int(scene_corners[2,0,1])),(int(scene_corners[3,0,0]), int(scene_corners[3,0,1])), (0,255,0), 4)
cv.line(res, (int(scene_corners[3,0,0]), int(scene_corners[3,0,1])),(int(scene_corners[0,0,0]), int(scene_corners[0,0,1])), (0,255,0), 4)
 
cv.imshow("end", res)

cv.waitKey()
