import mahotas as mh
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
import os
import joblib
import time

eps = 1e-7

# Feature extraction function which we used to create our csv files


def feature_extraction(orig_path, CAND, true_1, true_2, ss, model, save_dir, labels = ["SE", "EX", "NONE"]):
    
    orig = cv.imread(orig_path)[:,:,1]

    h_hsv = cv.cvtColor(cv.imread(orig_path), cv.COLOR_BGR2HSV)[:,:,0]

    try:
        true_1 = cv.imread(true_1, 0)
        true_1 = (true_1 > 0) * 255
        true_1 = np.array(true_1, dtype = np.uint8)
    except Exception as e:
        print(e)
    try:
        true_2 = cv.imread(true_2, 0)
        true_2 = (true_2 > 0) * 255
        true_2 = np.array(true_2, dtype = np.uint8)
    except Exception as e:
        print(e)
    try:
        print(true_1.shape, true_2.shape)
        print(set(list(true_1.flatten())))
        print(set(list(true_2.flatten())))
    except Exception as e:
        print(e)
       
    img = cv.imread(CAND, 0)
    contours,hierarchy = cv.findContours(img, 1, 2)
    contours = sorted(contours, key = lambda x: len(x), reverse = True)

    h, w = orig.shape
    result_se = np.zeros((h, w))
    result_ex = np.zeros((h, w))

    results_dict = {}

    classes = list(model.classes_)

    for i, j in enumerate(contours):

        mask = np.zeros(img.shape[:2], dtype="uint8")
        # loop over the contours
        cv.drawContours(mask, [j], -1, 255, -1)

        cnt = j

        M = cv.moments(cnt)
        x,y,w,h = cv.boundingRect(j)
        new = np.zeros(img.shape + (3,), dtype = 'uint8')
        new = cv.rectangle(new,(x,y),(x+w,y+h),(0,255,0),20)

        img_ = cv.bitwise_and(img, mask)

        #plt.imshow(np.hstack((img, mask, img_)))

        #plt.figure()
        A = img_[y:y+h,x:x+w]
        #plt.imshow(A)
        #plt.show()

        A = img_[y:y+h,x:x+w]
    
        A = np.array(A > 0, dtype = np.uint8) * 255

        glcm = greycomatrix(orig[y:y+h,x:x+w], [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        image_har = mh.features.haralick(orig[y:y+h,x:x+w]).mean(axis=0)
        hu = cv.HuMoments(cv.moments(A)).flatten()



        lbp = local_binary_pattern(orig[y:y+h,x:x+w], 24, 3, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins = 26, range=(0, 26))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        img_mean = orig[y:y+h,x:x+w].mean()

        h_mean = h_hsv[y:y+h, x:x+w].mean()


        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt,True)

        extra = [h_mean, area, perimeter, w, h]

        feature_vector = [greycoprops(glcm, 'energy')[0, 0], greycoprops(glcm, 'correlation')[0, 0], greycoprops(glcm, 'homogeneity')[0, 0], greycoprops(glcm, 'contrast')[0, 0], greycoprops(glcm, 'dissimilarity')[0, 0]] + list(image_har) + list(hu) + list(hist) + [img_mean] + extra
        feature_vector = np.array(feature_vector).reshape(1, -1)

        transformed = ss.transform(feature_vector)

        label = model.predict(transformed)
        probability = model.predict_proba(transformed)

        A = np.array(A / 255.0, dtype = np.float16)
        #print(A.max(), A.min())

        prob = classes.index(label)

        if labels.index(label) == 0:
            result_se[y:y+h,x:x+w] = A * probability[0][prob]
        elif labels.index(label) == 1:
            result_ex[y:y+h,x:x+w] = A * probability[0][prob]

    #print(results_dict)

    plt.imshow(result_se)
    plt.figure()
    plt.imshow(result_ex)
    plt.figure()
    try:
        plt.imshow(true_1)
        plt.figure()
    except Exception as e:
        print(e)
    try:
        plt.imshow(true_2)
    except Exception as e:
        print(e)
    #plt.show()

    # Saving reconstructions


    print(orig_path.split(os.path.sep)[-1].split(".")[0] + "_SE.npy")
    np.save(os.path.join(save_dir, orig_path.split(os.path.sep)[-1].split(".")[0] + "_SE.npy"), result_se)
    np.save(os.path.join(save_dir, orig_path.split(os.path.sep)[-1].split(".")[0] + "_EX.npy"), result_ex)

# Saving results


try:
    os.mkdir("Results")
except Exception as e:
    print(e)
try:
    os.mkdir("Results/SE_EX")
except Exception as e:
    print(e)

# Getting trained models from the Models folder

all_files = os.listdir("Models/SE_EX")
print(all_files)
models = [i for i in all_files if "_ss" not in i]

# Reconstructing test images from test candidates generated from image processing for all models

for mod in models:
    name = mod

    print(name)
    print(4)

    model = joblib.load(os.path.join("Models", "SE_EX", name))
    ss = joblib.load(os.path.join("Models", "SE_EX", "{}_ss.save".format("_".join(name.split("_")[0:2]))))
    res_dir = os.path.join("Results", "SE_EX", name)
    print(res_dir)
    os.mkdir(res_dir)
    all_test = os.listdir("test")

    imgs = [i.split(".")[0].split("_")[1] for i in all_test]

    for j, i in enumerate(imgs):
        print(j)
        print(i)
        x = feature_extraction("test/IDRiD_{}.jpg".format(i), "E:/Academic/MAIA/Semester_2/Main/ML_DL_AIA Project_Final/New Candidates/Exudates/SE-EX_Test/IDRiD_{}_Exudates final result.png".format(i), "groundtruths/test/soft exudates/IDRiD_{}_SE.tif".format(i), "groundtruths/test/hard exudates/IDRiD_{}_EX.tif".format(i), ss, model, res_dir)



