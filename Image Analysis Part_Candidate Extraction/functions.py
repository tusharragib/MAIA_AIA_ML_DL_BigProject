import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def gamma_correction(img,gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    result = cv.LUT(img, lookUpTable)
    return result

def sensitivity_area(img,grd,str):
    grd_gray=cv.cvtColor(grd,cv.COLOR_BGR2GRAY)
    grd_gray[grd_gray!=0]=255
    porc_descubiert=np.count_nonzero(cv.bitwise_and(img,grd_gray))/np.count_nonzero(grd_gray)
    area=np.count_nonzero(img)
    print("{}:  Porcent descubierto: {:.2f} y area cubierta: {}".format(str,porc_descubiert,area))
    return porc_descubiert,area

def focus_circle_mean(img_gray):
    result=img_gray.copy()
    circles = cv.HoughCircles(result, cv.HOUGH_GRADIENT, 2, 600,
                              param1=100, param2=60,
                              minRadius=1500, maxRadius=1900)
    print("cantidad de circulos:{}".format(len(circles)))
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print("circulos {}".format(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            mask = np.zeros_like(result)
            mask = cv.circle(mask, center, radius, 255, -1)
            prom = np.mean(result[mask == 255])
            #print(prom)
            result[mask == 0] = np.round(prom)
            return result



def optic_disk(img_gray):
    result=img_gray.copy()
    circles = cv.HoughCircles(result, cv.HOUGH_GRADIENT,2,600,
                              param1=100, param2=60,
                              minRadius=150, maxRadius=400)
    #print(len(circles))
    if circles is not None:
        circles = np.uint16(np.around(circles))
        #print(circles)
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            mask = np.zeros_like(result)
            mask = cv.circle(mask, center, radius, 255, -1)
            #imshow_ft("mascara",cv.circle(result, center, radius, 0, 4))
            return mask
    return 0

def focus_circle_mask(img_gray):
    result=img_gray.copy()
    circles = cv.HoughCircles(result, cv.HOUGH_GRADIENT,2,600,
                              param1=100, param2=60,
                              minRadius=1300, maxRadius=1900)
    #print(len(circles))
    if circles is not None:
        circles = np.uint16(np.around(circles))
        #print(circles)
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            mask = np.zeros_like(result)
            mask = cv.circle(mask, center, radius, 255, -1)
            #imshow_ft("mascara",cv.circle(result, center, radius, 0, 4))
            return mask
    return 0



def focus_circle_2_img(img_gray,img_repl,pixel_paint_outside):
    result=img_repl.copy()
    circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 2, 600,
                              param1=100, param2=60,
                              minRadius=1300, maxRadius=1900)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle outline
            radius = i[2]
            cv.circle(result, center, radius, 0, 2)

    #print(circles)
    center = (i[0], i[1])
    radio = i[2]
    mask = np.zeros_like(result)
    mask = cv.circle(mask, center, radio, 255-pixel_paint_outside, -1)
    # apply mask to image
    result = cv.bitwise_and(result, mask)
    return result

def focus_circle(img_gray,pixel_paint_outside):
    result=img_gray.copy()
    circles = cv.HoughCircles(result, cv.HOUGH_GRADIENT, 2, 500,
                              param1=100, param2=60,
                              minRadius=1300, maxRadius=1900)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle outline
            radius = i[2]
            #cv.circle(result, center, radius, 0, 5)
    #print(circles)
    center = (i[0], i[1])
    radio = i[2]
    mask = np.zeros_like(result)
    mask = cv.circle(mask, center, radio, 255-pixel_paint_outside, -1)
    #mask=cv.dilate(mask,cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))
    # apply mask to image
    result = cv.bitwise_and(result, mask)
    return result

def add_grdth(img, grd):
    img_col=img.copy()
    if len(img.shape)==2:
        img_col=cv.cvtColor(img_col,cv.COLOR_GRAY2BGR)
    return cv.addWeighted(img_col, 0.5, grd, 0.5, 0)

def substract_2_im(im1,im2):
    im1_16 = np.int16(im1)
    im2_16 = np.int16(im2)
    prep_img_16 = im1_16 - im2_16
    prep_img = np.uint8(cv.normalize(prep_img_16, 0, 255, norm_type=cv.NORM_MINMAX))
    return prep_img

def max_channel(bv_dif,grd_microaneu):#return max per channel
    mix_1 = cv.merge((bv_dif[:, :, 0], grd_microaneu[:, :, 0]))
    mix_2 = cv.merge((bv_dif[:, :, 1], grd_microaneu[:, :, 1]))
    mix_3 = cv.merge((bv_dif[:, :, 2], grd_microaneu[:, :, 2]))
    mix_1 = mix_1.max(axis=2)
    mix_2 = mix_2.max(axis=2)
    mix_3 = mix_3.max(axis=2)
    mix = cv.merge((mix_1, mix_2, mix_3))
    return mix

def sum_operation_se_different_directions(img,operation, width,height,n_se,norm=1):
    # create SEs
    base = np.zeros([width, width])
    k = int(width / 2 - height / 2)
    while k <= (width / 2 + height / 2):
        base = cv.line(base, (0, k), (width, k), 255)
        k = k + 1
        #print(k)
    SEs = []
    SEs.append(base)
    angle = 180.0 / n_se
    for k in range(1, n_se):
        SEs.append(cv.warpAffine(base, cv.getRotationMatrix2D((base.shape[0] / 2, base.shape[1] / 2), k * angle, 1.0),
                                 (width, width)))
    # cv.imshow("see",SEs[0])
    #print(SEs[0].shape)
    open_sum = np.uint16(0*cv.morphologyEx(img, operation, np.uint8(SEs[0])))
    i=0
    for se in SEs:
        open_sum += cv.morphologyEx(img, operation, np.uint8(se))
    if norm==1:
        result= cv.normalize(open_sum, 0, 255, norm_type=cv.NORM_MINMAX)
        return np.uint8(result)
    return open_sum

def sum_operation_se_different_directions_mean_std(img,operation, width,height,n_se):
    # create SEs
    base = np.zeros([width, width])
    k = int(width / 2 - height / 2)
    while k <= (width / 2 + height / 2):
        base = cv.line(base, (0, k), (width, k), 255)
        k = k + 1
        #print(k)
    SEs = []
    SEs.append(base)
    angle = 180.0 / n_se
    for k in range(1, n_se):
        SEs.append(cv.warpAffine(base, cv.getRotationMatrix2D((base.shape[0] / 2, base.shape[1] / 2), k * angle, 1.0),
                                 (width, width)))
    open_res = np.uint16(0*cv.morphologyEx(img, operation, np.uint8(SEs[0])))
    open_aux = np.uint16(0*cv.morphologyEx(img, operation, np.uint8(SEs[0])))
    f=1
    print(len(SEs))
    for se in SEs:
        if f==1:
            open_res = cv.morphologyEx(img, operation, np.uint8(se))
            f=0
        else:
            open_aux = cv.morphologyEx(img, operation, np.uint8(se))
            open_res = cv.merge((open_res, open_aux))
    open_sum=open_res.sum(axis=2)
    open_std=open_res.std(axis=2)
    sum_normalize= np.uint8(cv.normalize(np.float32(open_sum), 0, 255, norm_type=cv.NORM_MINMAX))
    std_normalize = np.uint8(cv.normalize(np.float32(open_std), 0, 255, norm_type=cv.NORM_MINMAX))
    return sum_normalize,std_normalize

def max_operation_se_different_directions(img,operation, width,height,n_se):
    # create SEs
    base = np.zeros([width, width])
    k = int(width / 2 - height / 2)
    while k <= (width / 2 + height / 2):
        base = cv.line(base, (0, k), (width, k), 255)
        k = k + 1
        #print(k)
    SEs = []
    SEs.append(base)
    angle = 180.0 / n_se
    for k in range(1, n_se):
        SEs.append(cv.warpAffine(base, cv.getRotationMatrix2D((base.shape[0] / 2, base.shape[1] / 2), k * angle, 1.0),
                                 (width, width)))
    # cv.imshow("see",SEs[0])
    #print(SEs[2].shape)
    open_max = cv.morphologyEx(img, operation, np.uint8(SEs[0]))
    open_max = cv.merge((open_max, open_max))
    for se in SEs:
        open_max[:, :, 1] = cv.morphologyEx(img, operation, np.uint8(se))
        open_max[:, :, 0] = open_max.max(axis=2)

    open_img = np.uint8(open_max[:, :, 0])
    return open_img

def min_operation_se_different_directions(img,operation, width,height,n_se):
    # create SEs
    base = np.zeros([width, width])
    k = int(width / 2 - height / 2)
    while k <= (width / 2 + height / 2):
        base = cv.line(base, (0, k), (width, k), 255)
        k = k + 1
        #print(k)
    SEs = []
    SEs.append(base)
    angle = 180.0 / n_se
    for k in range(1, n_se):
        SEs.append(cv.warpAffine(base, cv.getRotationMatrix2D((base.shape[0] / 2, base.shape[1] / 2), k * angle, 1.0),
                                 (width, width)))
    # cv.imshow("see",SEs[0])
    #print(SEs[2].shape)
    open_max = cv.morphologyEx(img, operation, np.uint8(SEs[0]))
    open_max = cv.merge((open_max, open_max))
    for se in SEs:
        open_max[:, :, 1] = cv.morphologyEx(img, operation, np.uint8(se))
        open_max[:, :, 0] = open_max.min(axis=2)

    open_img = np.uint8(open_max[:, :, 0])
    return open_img

def invert_image(imagem):
    imagem = (255-imagem)
    return imagem

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

def imshow_ft(str, img,norm=0):
    img_resized=img.copy()
    if norm==1:
        img_resized=np.uint8(cv.normalize(img_resized, 0, 255, norm_type=cv.NORM_MINMAX))
    img_resized=ResizeWithAspectRatio(img_resized, width=1280)
    cv.imshow(str, img_resized)


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def reconstruction_erode_dilations(img, struct_erode, dim_erode, struct_dil, dim_dil):
    img_erode = cv.erode(img, cv.getStructuringElement(struct_erode, dim_erode))
    marker_prev = img_erode.copy()
    marker_prevv = img_erode.copy()
    f = 1
    if len(img.shape)==3:
        while (cv.countNonZero(marker_prevv[:, :, 0] - marker_prev[:, :, 0]) != 0) | (
                cv.countNonZero(marker_prevv[:, :, 0] - marker_prev[:, :, 0]) != 0) | (
                cv.countNonZero(marker_prevv[:, :, 0] - marker_prev[:, :, 0]) != 0) | (f == 1):
            f = 0
            marker_prevv = marker_prev
            marker_prev = cv.dilate(marker_prev, cv.getStructuringElement(struct_dil, dim_dil))
            marker_prev = cv.min(marker_prev, img)
    else:
        while (cv.countNonZero(marker_prevv - marker_prev) != 0) | (f == 1):
            f = 0
            marker_prevv = marker_prev
            marker_prev = cv.dilate(marker_prev, cv.getStructuringElement(struct_dil, dim_dil))
            marker_prev = cv.min(marker_prev, img)
    return img_erode, marker_prev

def reconstruction_dilations(img_to_dilate, img, struct_dil, dim_dil):
    marker_prev = img_to_dilate.copy()
    marker_prevv = img_to_dilate.copy()
    f = 1
    if len(img.shape)==3:
        while (cv.countNonZero(marker_prevv[:, :, 0] - marker_prev[:, :, 0]) != 0) | (
                cv.countNonZero(marker_prevv[:, :, 0] - marker_prev[:, :, 0]) != 0) | (
                cv.countNonZero(marker_prevv[:, :, 0] - marker_prev[:, :, 0]) != 0) | (f == 1):
            f = 0
            marker_prevv = marker_prev
            marker_prev = cv.dilate(marker_prev, cv.getStructuringElement(struct_dil, dim_dil))
            marker_prev = cv.min(marker_prev, img)
    else:
        while (cv.countNonZero(marker_prevv - marker_prev) != 0) | (f == 1):
            f = 0
            marker_prevv = marker_prev
            marker_prev = cv.dilate(marker_prev, cv.getStructuringElement(struct_dil, dim_dil))
            marker_prev = cv.min(marker_prev, img)
    return marker_prev