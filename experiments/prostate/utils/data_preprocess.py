import SimpleITK as sitk
import numpy as np
import os

def normalize(arr):
    return (arr-np.mean(arr)) / np.std(arr)


clients = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
base_path = '../dataset/Prostate/data'
tar_path = '../dataset/Prostate/processed'
for client in clients:
    folder = os.path.join(base_path, client)
    nii_seg_list = [nii for nii in os.listdir(folder) if 'segmentation' in str(nii).lower()]
    slice_count = dict()
    tar_folder_path = os.path.join(tar_path,client)
    if not os.path.exists(tar_folder_path): os.makedirs(tar_folder_path)
    
    for nii_seg in nii_seg_list:
        nii_path = os.path.join(folder, nii_seg[:6]+'.nii.gz')
        nii_seg_path = os.path.join(folder, nii_seg)

        image_vol = sitk.ReadImage(nii_path)
        label_vol = sitk.ReadImage(nii_seg_path)
        image_vol = sitk.GetArrayFromImage(image_vol)
        label_vol = sitk.GetArrayFromImage(label_vol)
        label_vol[label_vol > 1] = 1
        has_label = list(set(np.where(label_vol>0)[0]))
        
        label_vol = label_vol[has_label]
        image_vol = image_vol[has_label]

        image_v3 = []
        for i in range(image_vol.shape[0]):
            if i==0:
                image = np.concatenate([np.expand_dims(image_vol[0, :, :],0),image_vol[i:i+2, :, :]],axis=0)
            elif i==image_vol.shape[0]-1:
                image = np.concatenate([image_vol[i-2:i, :, :],np.expand_dims(image_vol[i, :, :],0)])
            else:
                image = np.array(image_vol[i-1:i+2, :, :])

            image = np.transpose(image,(1,2,0))            
            assert image.shape == (384, 384,3)
            
            image_v3.append(image)
        image_v3 = np.asarray(image_v3)
        slice_count[nii_seg[:6]] = image_vol.shape[0]
        
        np.save(os.path.join(tar_folder_path,nii_seg[:6]+'.npy'),image_v3)
        np.save(os.path.join(tar_folder_path,nii_seg[:6]+'_segmentation.npy'),label_vol)
        
    

