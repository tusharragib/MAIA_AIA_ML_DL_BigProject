# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 23:22:15 2021

@author: tusha
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

def vis(path):
    try:
        img = np.load(path)
        plt.imshow(img, cmap = "gray")
        plt.show()
    except Exception as e:
        print(e)

#vis("E:/Academic/MAIA/Semester_2/Main/ML_DL_AIA Project_Final/Codes for Feature Extraction/Results/SE_EX/balanced_data_rf.save/IDRiD_55_EX.npy")
#vis("E:/Academic/MAIA/Semester_2/Main/ML_DL_AIA Project_Final/Codes for Feature Extraction/Results/SE_EX/balanced_data_rf.save/IDRiD_55_SE.npy")
vis("E:/Academic/MAIA/Semester_2/Main/ML_DL_AIA Project_Final/Codes for Feature Extraction/Results/MA_HE/balanced_data_rf.save/IDRiD_55_MA.npy")
vis("E:/Academic/MAIA/Semester_2/Main/ML_DL_AIA Project_Final/Codes for Feature Extraction/Results/MA_HE/balanced_data_rf.save/IDRiD_55_HE.npy")

