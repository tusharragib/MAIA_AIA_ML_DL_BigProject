import mahotas as mh
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
import os
import time

eps = 1e-7

# Used to extract features


def feature_extraction(orig_path, CAND, label_1, label_2, labels = ["SE", "EX"]):
    
    orig = cv.imread(orig_path)[:,:,1]

    # The HSV mean of the Hue channel will be used as a feature    

    h_hsv = cv.cvtColor(cv.imread(orig_path), cv.COLOR_BGR2HSV)[:,:,0]

    gt_1 = cv.imread(label_1, 0)
    gt_2 = cv.imread(label_2, 0)
    #print('gt_1', gt_1)
    #print('gt_2', gt_2)
    if ((gt_1 is not None) and (gt_2 is not None)):
       
        img = cv.imread(CAND, 0)
        contours,hierarchy = cv.findContours(img, 1, 2)
        contours = sorted(contours, key = lambda x: len(x), reverse = True)

        results_dict = {}

        # Looping through contours from biggest to smallest    

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


            A = img_[y:y+h,x:x+w]
        
            A = np.array(A > 0, dtype = np.uint8) * 255
            B = gt_1[y:y+h,x:x+w]
            B = np.array(B > 0, dtype = np.uint8) * 255
            inter_1 = cv.bitwise_and(A, B)
            C = gt_2[y:y+h,x:x+w]
            C = np.array(C > 0, dtype = np.uint8) * 255
            inter_2 = cv.bitwise_and(A, C)

            # Comparing with ground truth to check for a possible label


            sum1 = (inter_1 > 0).sum()
            sum2 = (B > 0).sum()
            sum3 = (inter_2 > 0).sum()
            sum4 = (C > 0).sum()

            sum5 = (A > 0).sum()

            # Calculating features


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

            extra = [h_mean, cx, cy, area, perimeter]

            try:
                if sum1/sum5 > 0.5:
                    results_dict[orig_path + "_{}_{}_{}_{}_{}".format(x, y, w, h, labels[0])] = [greycoprops(glcm, 'energy')[0, 0], greycoprops(glcm, 'correlation')[0, 0], greycoprops(glcm, 'homogeneity')[0, 0], greycoprops(glcm, 'contrast')[0, 0], greycoprops(glcm, 'dissimilarity')[0, 0]] + list(image_har) + list(hu) + list(hist) + [img_mean] + extra
                if sum3/sum5 > 0.5:
                    results_dict[orig_path + "_{}_{}_{}_{}_{}".format(x, y, w, h, labels[1])] = [greycoprops(glcm, 'energy')[0, 0], greycoprops(glcm, 'correlation')[0, 0], greycoprops(glcm, 'homogeneity')[0, 0], greycoprops(glcm, 'contrast')[0, 0], greycoprops(glcm, 'dissimilarity')[0, 0]] + list(image_har) + list(hu) + list(hist) + [img_mean] + extra
                else:
                    results_dict[orig_path + "_{}_{}_{}_{}_{}".format(x, y, w, h, "NONE")] = [greycoprops(glcm, 'energy')[0, 0], greycoprops(glcm, 'correlation')[0, 0], greycoprops(glcm, 'homogeneity')[0, 0], greycoprops(glcm, 'contrast')[0, 0], greycoprops(glcm, 'dissimilarity')[0, 0]] + list(image_har)  + list(hu) + list(hist) + [img_mean] + extra
            except Exception as e:
                print(e)
        print('EX+SE------------')
        
    # Checking for condition where this image has no ground truth
  

    elif ((gt_1 is None) and (gt_2 is None)):
        results_dict = {}
        print('No labels at all')

    # Checking for condition if only one ground truth exists

  
    elif (gt_1 is None):
        #gt_2 = cv.imread(label_2, 0)
        img = cv.imread(CAND, 0)
        contours,hierarchy = cv.findContours(img, 1, 2)
        contours = sorted(contours, key = lambda x: len(x), reverse = True)
        results_dict = {}
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


            A = img_[y:y+h,x:x+w]
            A = np.array(A > 0, dtype = np.uint8) * 255
            C = gt_2[y:y+h,x:x+w]
            C = np.array(C > 0, dtype = np.uint8) * 255
            inter_2 = cv.bitwise_and(A, C)
            sum3 = (inter_2 > 0).sum()
            sum4 = (C > 0).sum()
            sum5 = (A > 0).sum()
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

            extra = [h_mean, cx, cy, area, perimeter]


            try:
                if sum3/sum5 > 0.5:
                    results_dict[orig_path + "_{}_{}_{}_{}_{}".format(x, y, w, h, labels[1])] = [greycoprops(glcm, 'energy')[0, 0], greycoprops(glcm, 'correlation')[0, 0], greycoprops(glcm, 'homogeneity')[0, 0], greycoprops(glcm, 'contrast')[0, 0], greycoprops(glcm, 'dissimilarity')[0, 0]] + list(image_har)  + list(hu) + list(hist) + [img_mean]  + extra
                else:
                    results_dict[orig_path + "_{}_{}_{}_{}_{}".format(x, y, w, h, "NONE")] = [greycoprops(glcm, 'energy')[0, 0], greycoprops(glcm, 'correlation')[0, 0], greycoprops(glcm, 'homogeneity')[0, 0], greycoprops(glcm, 'contrast')[0, 0], greycoprops(glcm, 'dissimilarity')[0, 0]] + list(image_har)  + list(hu) + list(hist) + [img_mean] + extra
            except Exception as e:
                print(e)

        print('EX------------')

    # Checking for condition if only one ground truth exists

    elif (gt_2 is None):
        #gt_1 = cv.imread(label_1, 0)
        img = cv.imread(CAND, 0)
        contours,hierarchy = cv.findContours(img, 1, 2)
        contours = sorted(contours, key = lambda x: len(x), reverse = True)
        results_dict = {}

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


            A = img_[y:y+h,x:x+w]
            A = np.array(A > 0, dtype = np.uint8) * 255
            B = gt_1[y:y+h,x:x+w]
            B = np.array(B > 0, dtype = np.uint8) * 255
            inter_1 = cv.bitwise_and(A, B)


            sum1 = (inter_1 > 0).sum()
            sum2 = (B > 0).sum()
            sum5 = (A > 0).sum()

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

            extra = [h_mean, cx, cy, area, perimeter]

            try:
                if sum1/sum5 > 0.5:
                    results_dict[orig_path + "_{}_{}_{}_{}_{}".format(x, y, w, h, labels[0])] = [greycoprops(glcm, 'energy')[0, 0], greycoprops(glcm, 'correlation')[0, 0], greycoprops(glcm, 'homogeneity')[0, 0], greycoprops(glcm, 'contrast')[0, 0], greycoprops(glcm, 'dissimilarity')[0, 0]] + list(image_har)  + list(hu) + list(hist) + [img_mean]  + extra
                else:
                    results_dict[orig_path + "_{}_{}_{}_{}_{}".format(x, y, w, h, "NONE")] = [greycoprops(glcm, 'energy')[0, 0], greycoprops(glcm, 'correlation')[0, 0], greycoprops(glcm, 'homogeneity')[0, 0], greycoprops(glcm, 'contrast')[0, 0], greycoprops(glcm, 'dissimilarity')[0, 0]] + list(image_har)  + list(hu) + list(hist) + [img_mean] + extra
            except Exception as e:
                print(e)

        print('SE------------')
    


    print(results_dict)

    
    return results_dict




import pandas as pd

# Converting to Data Frame


df = pd.DataFrame({"name": [], "feature_1": [], "feature_2": [], "feature_3": [], "feature_4": [], "feature_5": [], 
                   "feature_har_1": [], "feature_har_2": [], "feature_har_3": [], "feature_har_4": [], "feature_har_5": [], 
                   "feature_har_6": [], "feature_har_7": [], "feature_har_8": [], "feature_har_9": [], "feature_har_10": [], 
                   "feature_har_11": [], "feature_har_12": [], "feature_har_13": [], "feature_hu_1": [], "feature_hu_2": [],
                   "feature_hu_3": [], "feature_hu_4": [], "feature_hu_5": [], "feature_hu_6": [], "feature_hu_7": [],
                   "hist_lbp_0": [], "hist_lbp_1": [], "hist_lbp_2": [], "hist_lbp_3": [], "hist_lbp_4": [], "hist_lbp_5": [], 
                   "hist_lbp_6": [], "hist_lbp_7": [], "hist_lbp_8": [], "hist_lbp_9": [], "hist_lbp_10": [], "hist_lbp_11": [], 
                   "hist_lbp_12": [], "hist_lbp_13": [], "hist_lbp_14": [], "hist_lbp_15": [], "hist_lbp_16": [], "hist_lbp_17": [],
                   "hist_lbp_18": [], "hist_lbp_19": [], "hist_lbp_20": [], "hist_lbp_21": [], "hist_lbp_22": [], "hist_lbp_23": [],
                   "hist_lbp_24": [], "hist_lbp_25": [], "img_mean": [], "img_h_mean": [], "cx": [], "cy": [], "area": [], "perimeter": []})


                   
###image = cv.imread('C:/Users/HP/Retinal/training/IDRiD_01.jpg')
#bloodvessel = extract_bv(image)
#cv2.imwrite('C:/Users/HP/Retinal/training/Base12Bloodvessels/' + "_bloodvessel.png",bloodvessel)

start = time.time()

pathFolder = "E:/Academic/MAIA/Semester_2/Main/ML_DL_AIA Project_Final/Train_S"
pathFolder_handCrafted = 'E:/Academic/MAIA/Semester_2/Main/ML_DL_AIA Project_Final/New Candidates/Exudates/SE-EX_Training/'
pathFolderSE = 'E:/Academic/MAIA/Semester_2/Main/ML_DL_AIA Project_Final/GT/training/soft exudates/'
pathFolderEX = 'E:/Academic/MAIA/Semester_2/Main/ML_DL_AIA Project_Final/GT/training/hard exudates/'
filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
print(filesArray)

count = 1

for file_name in filesArray:
    print(count)
    count += 1
    file_name_no_extension = os.path.splitext(file_name)[0]
    print(file_name)
    print(pathFolderEX +file_name_no_extension+"_EX.tif")
    print(pathFolderSE +file_name_no_extension+"_SE.tif")
    print(pathFolder_handCrafted + file_name_no_extension + '_Exudates final result.png')
    features = feature_extraction(pathFolder +'/'+ file_name, 
                                  pathFolder_handCrafted + file_name_no_extension + '_Exudates final result.png', 
                                  pathFolderSE +file_name_no_extension+"_SE.tif", 
                                  pathFolderEX +file_name_no_extension+"_EX.tif")
    print(file_name)

    print(len(features))
    
    #if (feature_list_artificial ):
    
    if not features:
        print('Out of loop')
        continue
        
    for j in list(features.items()):
          df = df.append({"name": j[0], "feature_1": j[1][0], "feature_2": j[1][1], "feature_3": j[1][2], "feature_4": j[1][3], "feature_5": j[1][4],
              "feature_har_1": j[1][5], "feature_har_2": j[1][6], "feature_har_3": j[1][7], "feature_har_4": j[1][8], "feature_har_5": j[1][9], 
              "feature_har_6": j[1][10], "feature_har_7": j[1][11], "feature_har_8": j[1][12], "feature_har_9": j[1][13], "feature_har_10": j[1][14], 
              "feature_har_11": j[1][15], "feature_har_12": j[1][16], "feature_har_13": j[1][17], "feature_hu_1": j[1][18], "feature_hu_2": j[1][19],
              "feature_hu_3": j[1][20], "feature_hu_4": j[1][21], "feature_hu_5": j[1][22], "feature_hu_6": j[1][23], "feature_hu_7": j[1][24],
              "hist_lbp_0": j[1][25], "hist_lbp_1": j[1][26], "hist_lbp_2": j[1][27], "hist_lbp_3": j[1][28], "hist_lbp_4": j[1][29], "hist_lbp_5": j[1][30], 
              "hist_lbp_6": j[1][31], "hist_lbp_7": j[1][32], "hist_lbp_8": j[1][33], "hist_lbp_9": j[1][34], "hist_lbp_10": j[1][35], "hist_lbp_11": j[1][36], 
              "hist_lbp_12": j[1][37], "hist_lbp_13": j[1][38], "hist_lbp_14": j[1][39], "hist_lbp_15": j[1][40], "hist_lbp_16": j[1][41], "hist_lbp_17": j[1][42],
              "hist_lbp_18": j[1][43], "hist_lbp_19": j[1][44], "hist_lbp_20": j[1][45], "hist_lbp_21": j[1][46], "hist_lbp_22": j[1][47], "hist_lbp_23": j[1][48],
              "hist_lbp_24": j[1][49], "hist_lbp_25": j[1][50], "img_mean": j[1][51], "img_h_mean": j[1][52], "cx": j[1][53], "cy": j[1][54], "area": j[1][55], 
              "perimeter": j[1][56]}, ignore_index = True)

    # elif (len(feature_list_artificial) > 2):
    #     feature_list_artificial = [features, features]
    #     for i in feature_list_artificial:
    #         for j in list(i.items()):
    #             df = df.append({"name": j[0], "feature_1": j[1][0], "feature_2": j[1][1], "feature_3": j[1][2], "feature_4": j[1][3], "feature_5": j[1][4],
    #             "feature_har_1": j[1][5], "feature_har_2": j[1][6], "feature_har_3": j[1][7], "feature_har_4": j[1][8], "feature_har_5": j[1][9], 
    #             "feature_har_6": j[1][10], 


print(time.time() - start)

df["Label"] = df["name"].apply(lambda x: x.split("_")[-1])

try:
  os.mkdir("featuressssssss_3")
except:
  pass
#df.to_csv(pathFolder+'/'+'test_fetures_extraction_SE_EX.csv', index = False)
df.to_csv('featuressssssss_3/train_fetures_extraction_SE_EX.csv', index = False)

print(df.shape)
