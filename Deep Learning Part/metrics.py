import numpy as np
from sklearn.metrics import precision_recall_fscore_support, auc, precision_recall_curve
import matplotlib.pyplot as plt

def calculate_plot_aucpr_ind(auc_full_mask, auc_full_output, str='!'):
    print(auc_full_mask.flatten().shape)
    print(auc_full_output.flatten().shape)
    precision, recall, th = precision_recall_curve(auc_full_mask.flatten(), auc_full_output.flatten())
    print(recall.shape)
    plt.figure()
    plt.plot(recall,precision)
    plt.title(f'recall_{str}')
    plt.show()
    print(f'AUC Precision Recall {str}: {auc(recall, precision)}')

with open('auc_mask_4_train5_14epochs-001.npy', 'rb') as f:
    auc_full_mask = np.load(f)
with open('auc_output_4_train5_14epochs-002.npy', 'rb') as f:
    auc_full_output = np.load(f)

calculate_plot_aucpr_ind(auc_full_mask[:,0,:,:],auc_full_output[:,0,:,:],'MA')
print("siguiente")
calculate_plot_aucpr_ind(auc_full_mask[:,1,:,:],auc_full_output[:,1,:,:],'HE')
print("siguiente")
calculate_plot_aucpr_ind(auc_full_mask[:,2,:,:],auc_full_output[:,2,:,:],'EX')
print("siguiente")
calculate_plot_aucpr_ind(auc_full_mask[:,3,:,:],auc_full_output[:,3,:,:],'SE')
