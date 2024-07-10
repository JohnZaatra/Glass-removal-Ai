import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import pandas as pd
import numpy as np

class EyeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe  # CSV file as a dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        ocular_treatment_id = self.dataframe.iloc[idx]['Ocular Treatment ID']
        last_efficacy_index = self.dataframe.iloc[idx]['Last_UCVA']
        anterior_image, posterior_image = self.load_images(ocular_treatment_id)
        
        # Concatenate images side by side (width-wise)
        combined_image = Image.new('RGB', (anterior_image.width + posterior_image.width, anterior_image.height))
        combined_image.paste(anterior_image, (0, 0))
        combined_image.paste(posterior_image, (anterior_image.width, 0))
        
        if self.transform:
            combined_image = self.transform(combined_image)
        
        # Process tabular data to generate mlp_input_tensor
        tabular_data = self.dataframe.iloc[idx].drop(['Ocular Treatment ID', 'Last_UCVA', 'File'])
        mlp_input_tensor = torch.tensor(tabular_data.values.astype(np.float32))

        return combined_image, mlp_input_tensor, torch.tensor(last_efficacy_index, dtype=torch.float)

    def load_images(self, ocular_treatment_id):
        right_eye_dir = "/home/janz/PROJECT/Eye_Scans_Data/op2_png"
        left_eye_dir = "/home/janz/PROJECT/Eye_Scans_Data/op5_png"

        id_code, eye, _, _ = ocular_treatment_id.split('-')
        eye_dir = right_eye_dir if eye == 'Right' else left_eye_dir
        anterior_image_path = glob.glob(os.path.join(eye_dir, f"{id_code}_*_Tangential_anterior.png"))[0]
        posterior_image_path = glob.glob(os.path.join(eye_dir, f"{id_code}_*_Tangential_posterior.png"))[0]
        anterior_image = Image.open(anterior_image_path).convert('RGB')
        posterior_image = Image.open(posterior_image_path).convert('RGB')
        return anterior_image, posterior_image
