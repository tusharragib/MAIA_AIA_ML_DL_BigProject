import numpy as np
from sklearn.metrics import precision_recall_fscore_support, auc, precision_recall_curve
import matplotlib.pyplot as plt
import cv2 as cv
import os

def calculate_plot_aucpr_ind(auc_full_mask, auc_full_output, str='!', save_path = ""):
    x = auc_full_mask.flatten()
    print(x.shape, x.dtype)
    y = auc_full_output.flatten()
    print(y.shape, y.dtype)
    precision, recall, th = precision_recall_curve(x, y)
    print(recall.shape)
    plt.figure()
    plt.plot(recall,precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f'recall_{str}')
    #plt.show()
    plt.savefig(save_path)
    print(f'AUC Precision Recall {str}: {auc(recall, precision)}')
    return auc(recall, precision)

# with open('auc_mask_4_train5_14epochs-001.npy', 'rb') as f:
#     auc_full_mask = np.load(f)
# with open('auc_output_4_train5_14epochs-002.npy', 'rb') as f:
#     auc_full_output = np.load(f)

# calculate_plot_aucpr_ind(auc_full_mask[:,0,:,:],auc_full_output[:,0,:,:],'MA')
# print("siguiente")
# calculate_plot_aucpr_ind(auc_full_mask[:,1,:,:],auc_full_output[:,1,:,:],'HE')
# print("siguiente")
# calculate_plot_aucpr_ind(auc_full_mask[:,2,:,:],auc_full_output[:,2,:,:],'EX')
# print("siguiente")
# calculate_plot_aucpr_ind(auc_full_mask[:,3,:,:],auc_full_output[:,3,:,:],'SE')

try:
    os.mkdir("Figures")
except Exception as e:
    print(e)

MA = []
HE = []

folders = os.listdir(os.path.join("Results", "MA_HE"))

for folder in folders:
    imgs = os.listdir("groundtruths/test/microaneurysms")

    A = np.array([], dtype = np.float16)
    B = np.array([], dtype = np.bool_)

    print(len(imgs))

    for j, i in enumerate(imgs):
        print(j)
        a = np.load(os.path.join("Results", "MA_HE", folder, "{}.npy".format(i.split(".")[0])))
        b = cv.imread("groundtruths/test/microaneurysms/{}".format(i), 0)
        b = b > 0
        b = np.array(b, dtype = np.bool_)
        a = np.array(a, dtype = np.float16)
        A = np.append(A, a)
        A.dtype = np.float16
        B = np.append(B, b)
        B.dtype = np.bool_
        print(B.dtype, A.dtype)
        print(A.shape)
        print(B.shape)

    ma = calculate_plot_aucpr_ind(B.flatten(),A.flatten(), 'MA', "Figures/{}_MA.png".format(folder))

    MA.append(folder + "_" + str(ma))

    imgs = os.listdir("groundtruths/test/haemorrhages")

    A = np.array([], dtype = np.float16)
    B = np.array([], dtype = np.bool_)

    print(len(imgs))

    for j, i in enumerate(imgs):
        print(j)
        a = np.load(os.path.join("Results", "MA_HE", folder, "{}.npy".format(i.split(".")[0])))
        b = cv.imread("groundtruths/test/haemorrhages/{}".format(i), 0)
        b = b > 0
        b = np.array(b, dtype = np.bool_)
        a = np.array(a, dtype = np.float16)
        A = np.append(A, a)
        A.dtype = np.float16
        B = np.append(B, b)
        B.dtype = np.bool_
        print(B.dtype, A.dtype)
        print(A.shape)
        print(B.shape)

    he = calculate_plot_aucpr_ind(B.flatten(),A.flatten(), 'HE', "Figures/{}_HE.png".format(folder))

    HE.append(folder + "_" + str(he))

SE = []
EX = []

folders = os.listdir(os.path.join("Results", "SE_EX"))

for folder in folders:
    imgs = os.listdir("groundtruths/test/soft exudates")

    A = np.array([], dtype = np.float16)
    B = np.array([], dtype = np.bool_)

    print(len(imgs))

    for j, i in enumerate(imgs):
        print(j)
        a = np.load(os.path.join("Results", "SE_EX", folder, "{}.npy".format(i.split(".")[0])))
        b = cv.imread("groundtruths/test/soft exudates/{}".format(i), 0)
        b = b > 0
        b = np.array(b, dtype = np.bool_)
        a = np.array(a, dtype = np.float16)
        A = np.append(A, a)
        A.dtype = np.float16
        B = np.append(B, b)
        B.dtype = np.bool_
        print(B.dtype, A.dtype)
        print(A.shape)
        print(B.shape)

    se = calculate_plot_aucpr_ind(B.flatten(),A.flatten(), 'SE', "Figures/{}_SE.png".format(folder))

    SE.append(folder + "_" + str(se))

    imgs = os.listdir("groundtruths/test/hard exudates")

    A = np.array([], dtype = np.float16)
    B = np.array([], dtype = np.bool_)

    print(len(imgs))

    for j, i in enumerate(imgs):
        print(j)
        a = np.load(os.path.join("Results", "SE_EX", folder, "{}.npy".format(i.split(".")[0])))
        b = cv.imread("groundtruths/test/hard exudates/{}".format(i), 0)
        b = b > 0
        b = np.array(b, dtype = np.bool_)
        a = np.array(a, dtype = np.float16)
        A = np.append(A, a)
        A.dtype = np.float16
        B = np.append(B, b)
        B.dtype = np.bool_
        print(B.dtype, A.dtype)
        print(A.shape)
        print(B.shape)

    ex = calculate_plot_aucpr_ind(B.flatten(),A.flatten(), 'EX', "Figures/{}_EX.png".format(folder))

    EX.append(folder + "_" + str(ex))

print(MA)
print(HE)
print(SE)
print(EX)

with open("MA_Result.txt", "w") as f:
    f.write("\n".join(MA))

with open("HE_Result.txt", "w") as f:
    f.write("\n".join(HE))

with open("SE_Result.txt", "w") as f:
    f.write("\n".join(SE))

with open("EX_Result.txt", "w") as f:
    f.write("\n".join(EX))