from collections import Counter
import argparse
import cv2
import kociemba
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as sp
from sklearn.cluster import KMeans

print('Program RUBIK PIXEL ART start . . .')
print('ต้องการรูปกี่ block')
blockimg = int(input())
# ต้องการภาพ 6 ด้านของรูบิค มาทำ obj detect และเก็บสี และอีก1ภาพที่จำทำเป็ด้านของรูบิค pixel art และเป็นสีที่เก็บจากรูบิค

# import รูป rubik แต่ละหน้า
FstSide = cv2.imread('/Users/chomusuke/Desktop/RubikProjectPic/FstSideimg.jpg')
SndSide = cv2.imread('/Users/chomusuke/Desktop/RubikProjectPic/SndSideimg.jpg')
TrdSide = cv2.imread('/Users/chomusuke/Desktop/RubikProjectPic/TrdSideimg.jpg')
FthSide = cv2.imread('/Users/chomusuke/Desktop/RubikProjectPic/FthSideimg.jpg')
FfhSide = cv2.imread('/Users/chomusuke/Desktop/RubikProjectPic/FfhSideimg.jpg')
SthSide = cv2.imread('/Users/chomusuke/Desktop/RubikProjectPic/SthSideimg.jpg')

# รูป preprocess H*W*d
img1 = cv2.imread('/Users/chomusuke/Desktop/RubikProjectPic/e2.jpg')
print('dimension ของรูป original = ', img1.shape[0], img1.shape[1])

# resize ก่อนหาค่า mod ไปลบ จะได้หาค่าของตารางได้ที่เอาไมเป็น pixel แต่ละหน่วยของหน้ารูบิค
# percent_sclae = 5 #อยากได้กี่เปอเซ็นใส่ตรงนี้
# height_resize = int(img1.shape[0] * percent_sclae / 100)
# width_resize = int(img1.shape[1] * percent_sclae / 100)
# dsize = (  width_resize,height_resize)
# output1 = cv2.resize(img1, dsize)
# cv2.imwrite('/Users/chomusuke/Desktop/RubikProjectPic/OrgImgResizeed.bmp',output1) #เขียนเป็นรูปให่ไม่เขียนทับ เพราะกลัวว่าจะ a liitle bit buggy
# img2 = cv2.imread('/Users/chomusuke/Desktop/RubikProjectPic/OrgImgResizeed.bmp')
# # cv2.imshow("masked", img2)
# # cv2.waitKey(0)
# print('dimension หลังจาก resize = ', img2.shape[0] ,img2.shape[1] )

# ลด sclae ของรูป mod ด้วย 3 แล้วเอาไปลบ Widht Hight เพื่อได้เท่ากับรูบิคหน้านึงหารลงตัวทุกพิกเซล
new_himg = int(img1.shape[0] % 3)
new_mimg = int(img1.shape[1] % 3)
new_dimg = int(img1.shape[2] % 3)
print('ค่า shape mod 3 ของ w h = ', new_himg, new_mimg)
height_moded = int(img1.shape[0] - new_himg)
width_moded = int(img1.shape[1] - new_mimg)
msize = (width_moded, height_moded)
output2 = cv2.resize(img1, msize)
cv2.imwrite('/Users/chomusuke/Desktop/RubikProjectPic/OrgImgResizeed.bmp', output2)
# เขียนเป็นรูปให่ไม่เขียนทับ เพราะกลัวว่าจะ a liitle bit buggy
img1 = cv2.imread('/Users/chomusuke/Desktop/RubikProjectPic/OrgImgResizeed.bmp')
print('dimension หลังจาก mod3 = ', img1.shape[0], img1.shape[1])


# ค่า array ทั้งหมดของรูปหลังจากการ resize
# image_data = np.asarray(img2)
# for i in range(len(image_data)):
#     for j in range(len(image_data[0])):
#
#         print(image_data[i][j])

# 1
def FirstSide():
    height_resize = int(FstSide.shape[0] * 10 / 100)
    width_resize = int(FstSide.shape[1] * 10 / 100)
    dsize = (width_resize, height_resize)
    FstSideResized = cv2.resize(FstSide, dsize)

    modified_image = cv2.resize(FstSideResized, (FstSideResized.shape[0], FstSideResized.shape[1]),
                                interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    clf = KMeans(n_clusters=2)  # no. of col
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    ordered_colors1 = ordered_colors[1]
    b1, g1, r1 = ordered_colors1

    print("ด้านที่ 1 คือสี r: ", int(r1), 'g: ', int(g1), 'b: ', int(b1))

    img = FstSideResized
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    (ret, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edge = cv2.Canny(thresh, 100, 200)
    (cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total = 0
    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(res2, [approx], -1, (0, 255, 0), 4)
        total += 1
    # print ("I found {0} RET in that image".format(total))
    cv2.imshow("OutputS1", res2)

    return r1, g1, b1


# 2
def SecondSide():
    height_resize = int(SndSide.shape[0] * 10 / 100)
    width_resize = int(SndSide.shape[1] * 10 / 100)
    dsize = (width_resize, height_resize)
    SndSideResized = cv2.resize(SndSide, dsize)

    modified_image = cv2.resize(SndSideResized, (SndSideResized.shape[0], SndSideResized.shape[1]),
                                interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    clf = KMeans(n_clusters=2)  # no. of col
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    ordered_colors2 = ordered_colors[1]
    b2, g2, r2 = ordered_colors2

    print("ด้านที่ 2 คือสี r: ", int(r2), 'g: ', int(g2), 'b: ', int(b2))

    img = SndSideResized
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    (ret, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edge = cv2.Canny(thresh, 100, 200)
    (cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total = 0
    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(res2, [approx], -1, (0, 255, 0), 4)
        total += 1
    # print ("I found {0} RET in that image".format(total))
    cv2.imshow("OutputS2", res2)

    return r2, g2, b2


# 3
def ThirdSide():
    height_resize = int(TrdSide.shape[0] * 10 / 100)
    width_resize = int(TrdSide.shape[1] * 10 / 100)
    dsize = (width_resize, height_resize)
    TrdSideResized = cv2.resize(TrdSide, dsize)

    modified_image = cv2.resize(TrdSideResized, (TrdSideResized.shape[0], TrdSideResized.shape[1]),
                                interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    clf = KMeans(n_clusters=2)  # no. of col
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    ordered_colors3 = ordered_colors[1]
    b3, g3, r3 = ordered_colors3

    print("ด้านที่ 3 คือสี r: ", int(r3), 'g: ', int(g3), 'b: ', int(b3))

    img = TrdSideResized
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    (ret, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edge = cv2.Canny(thresh, 100, 200)
    (cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total = 0
    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(res2, [approx], -1, (0, 255, 0), 4)
        total += 1
    # print ("I found {0} RET in that image".format(total))
    cv2.imshow("OutputS3", res2)

    return r3, g3, b3


# 4
def FouthSide():
    height_resize = int(FthSide.shape[0] * 10 / 100)
    width_resize = int(FthSide.shape[1] * 10 / 100)
    dsize = (width_resize, height_resize)
    FthSideResized = cv2.resize(FthSide, dsize)

    modified_image = cv2.resize(FthSideResized, (FthSideResized.shape[0], FthSideResized.shape[1]),
                                interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    clf = KMeans(n_clusters=2)  # no. of col
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    ordered_colors4 = ordered_colors[1]
    b4, g4, r4 = ordered_colors4

    print("ด้านที่ 4 คือสี r: ", int(r4), 'g: ', int(g4), 'b: ', int(b4))

    img = FthSideResized
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    (ret, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edge = cv2.Canny(thresh, 100, 200)
    (cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total = 0
    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(res2, [approx], -1, (0, 255, 0), 4)
        total += 1
    # print ("I found {0} RET in that image".format(total))
    cv2.imshow("OutputS4", res2)

    return r4, g4, b4


# 5
def FifthSide():
    height_resize = int(FfhSide.shape[0] * 10 / 100)
    width_resize = int(FfhSide.shape[1] * 10 / 100)
    dsize = (width_resize, height_resize)
    FfhSideResized = cv2.resize(FfhSide, dsize)

    modified_image = cv2.resize(FfhSideResized, (FfhSideResized.shape[0], FfhSideResized.shape[1]),
                                interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    clf = KMeans(n_clusters=2)  # no. of col
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    ordered_colors5 = ordered_colors[1]
    b5, g5, r5 = ordered_colors5

    print("ด้านที่ 5 คือสี r: ", int(r5), 'g: ', int(g5), 'b: ', int(b5))

    img = FfhSideResized
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    (ret, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edge = cv2.Canny(thresh, 100, 200)
    (cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total = 0
    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(res2, [approx], -1, (0, 255, 0), 4)
        total += 1
    # print ("I found {0} RET in that image".format(total))
    cv2.imshow("OutputS5", res2)

    return r5, g5, b5


# 6
def SixthSide():
    height_resize = int(SthSide.shape[0] * 10 / 100)
    width_resize = int(SthSide.shape[1] * 10 / 100)
    dsize = (width_resize, height_resize)
    SthSideResized = cv2.resize(SthSide, dsize)

    modified_image = cv2.resize(SthSideResized, (SthSideResized.shape[0], SthSideResized.shape[1]),
                                interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    clf = KMeans(n_clusters=2)  # no. of col
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    ordered_colors6 = ordered_colors[1]
    b6, g6, r6 = ordered_colors6

    print("ด้านที่ 6 คือสี r: ", int(r6), 'g: ', int(g6), 'b: ', int(b6))

    img = SthSideResized
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    # cv2.imshow('res2',res2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    (ret, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edge = cv2.Canny(thresh, 100, 200)
    (cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total = 0
    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        cv2.drawContours(res2, [approx], -1, (0, 255, 0), 4)
        total += 1
    # print ("I found {0} RET in that image".format(total))
    cv2.imshow("OutputS6", res2)

    return r6, g6, b6


# หาสีในแต่ละหน้า ด้วยการ cluster detect และ  bind สีที่จะทำ image pixelation
# FirstSide()
# SecondSide()
# ThirdSide()
# FouthSide()
# FifthSide()
# SixthSide()

# ------------------------------------------------------------------------------------------------------

# img = img1
# Z = img.reshape((-1,3))
# # convert to np.float32
# Z = np.float32(Z)
# # define criteria, number of clusters(K) and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 6
# ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# # Now convert back into uint8, and make original image
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((img.shape))
# cv2.imshow('res2',res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# height, width = res2.shape[:2]
#
# # Resize input to "pixelated" size
# h = 64
# w = 64
# temp = cv2.resize(res2, (h, w), interpolation=cv2.INTER_LINEAR)
# # Initialize output image
#
# cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
# img1 = temp

colorpalte1 = FirstSide()
colorpalte2 = SecondSide()
colorpalte3 = ThirdSide()
colorpalte4 = FouthSide()
colorpalte5 = FifthSide()
colorpalte6 = SixthSide()
cv2.waitKey(0)

# contrasting
# -----Converting image to LAB Color model-----------------------------------
lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
cv2.imshow("lab", lab)

# -----Splitting the LAB image to different channels-------------------------
l, a, b = cv2.split(lab)
cv2.imshow('l_channel', l)
cv2.imshow('a_channel', a)
cv2.imshow('b_channel', b)

# -----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
cv2.imshow('CLAHE output', cl)

# -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl, a, b))
cv2.imshow('limg', limg)

# -----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imshow('final', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
# _____END_____#
temp = final
#  ทำให้รูป sharpen
# smoothed = cv2.GaussianBlur(temp, (9, 9), 10)
# temp = cv2.addWeighted(temp, 1.5, smoothed, -0.5, 0)
# ทำรูป
h = blockimg
w = blockimg
temp = cv2.resize(temp, (h, w), interpolation=cv2.INTER_LINEAR)
# Initialize output image
cv2.resize(temp, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)
cv2.imshow("blocking img before bind color ", temp)

main_colors = [(colorpalte1),
               (colorpalte2),
               (colorpalte3),
               (colorpalte4),
               (colorpalte5),
               (colorpalte6)
               ]

print('main colors = ', main_colors)

img1 = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
h, w, bpp = np.shape(img1)

for py in range(0, h):
    for px in range(0, w):
        ########################
        # หาค่าสีใกล้เคียงกับที่ bing ค่าที่สุด
        # reference : https://stackoverflow.com/a/22478139/9799700
        input_color = (img1[py][px][0], img1[py][px][1], img1[py][px][2])
        tree = sp.KDTree(main_colors)
        ditsance, result = tree.query(input_color)
        nearest_color = main_colors[result]
        ###################

        img1[py][px][0] = nearest_color[0]
        img1[py][px][1] = nearest_color[1]
        img1[py][px][2] = nearest_color[2]

# show image
plt.figure()
plt.axis("off")
plt.imshow(img1)
plt.show()

countH = img1.shape[0]
countW = img1.shape[1]
rubiks_count = int((countH * countW) / 9)
print('ต้องการรูบิค = ', rubiks_count, 'ลูก')
