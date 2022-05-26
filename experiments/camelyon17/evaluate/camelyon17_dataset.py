import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class CamelyonDataset(Dataset):
    def __init__(self, root_dir, transform, split):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(CamelyonDataset, self).__init__()
        self.data_dir = '/research/pheng4/qdliu/hzyang/test_time_medical/dataset/camelyon17_v1.0/'
        self.original_resolution = (96,96)

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'})

        # Get the y values
        self.y_array = torch.LongTensor(self.metadata_df['tumor'].values)
        self.y_size = 1
        self.n_classes = 2

        # Get filenames
        self.input_array = [
            f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            for patient, node, x, y in
            self.metadata_df.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)]

        # Extract splits
        # Note that the hospital numbering here is different from what's in the paper,
        # where to avoid confusing readers we used a 1-indexed scheme and just labeled the test hospital as 5.
        # Here, the numbers are 0-indexed.
        test_center = 2
        val_center = 1

        self.split_dict = {
            'train': 0,
            'id_val': 1,
            'test': 2,
            'val': 3
        }
        self.split_names = {
            'train': 'Train',
            'id_val': 'Validation (ID)',
            'test': 'Test',
            'val': 'Validation (OOD)',
        }
        centers = self.metadata_df['center'].values.astype('long')
        num_centers = int(np.max(centers)) + 1
        val_center_mask = (self.metadata_df['center'] == val_center)
        test_center_mask = (self.metadata_df['center'] == test_center)
        self.metadata_df.loc[val_center_mask, 'split'] = self.split_dict['val']
        self.metadata_df.loc[test_center_mask, 'split'] = self.split_dict['test']
        '''
        self._split_scheme = split_scheme
        if self._split_scheme == 'official':
            pass
        elif self._split_scheme == 'in-dist':
            # For the in-distribution oracle,
            # we move slide 23 (corresponding to patient 042, node 3 in the original dataset)
            # from the test set to the training set
            slide_mask = (self._metadata_df['slide'] == 23)
            self._metadata_df.loc[slide_mask, 'split'] = self.split_dict['train']
        else:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        '''
        self.split_array = self.metadata_df['split'].values
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]
        self.y_array = self.y_array[split_idx]
        print(split_idx)
        tmp = []
        for idx in split_idx:
            tmp.append(self.input_array[idx]) 
        
        self.input_array = tmp #self.input_array[split_idx]
        '''
        self.metadata_array = torch.stack(
            (torch.LongTensor(centers),
             torch.LongTensor(self.metadata_df['slide'].values),
             self.y_array),
            dim=1)
        self.metadata_fields = ['hospital', 'slide', 'y']
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['slide'])
        '''
        self.transform = transform

        print('Total # images:{}, labels:{}'.format(len(self.input_array),len(self.y_array)))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        img_filename = os.path.join(
           self.data_dir,
           self.input_array[index])
        x = Image.open(img_filename).convert('RGB')
        y = self.y_array[index]
        #y = F.one_hot(y, num_classes=2)
        if self.transform is not None:
            x = self.transform(x)
        return  x, y

    def __len__(self):
        return len(self.y_array)


