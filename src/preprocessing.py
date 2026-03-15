#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Colab \\Notebooks/microscopy_self_supervised_learning/microscopy_self_supervised_learning')


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# In[4]:


metadata_with_moa = pd.read_csv("data/processed/metadata_with_moa.csv")
metadata_with_moa.head(3)


# Step 1: Load & stack channels - Merging channels for faster traianing time

# In[19]:


import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.transform import resize

BASE_DIR = "/content/drive/MyDrive/Colab Notebooks/microscopy_self_supervised_learning/microscopy_self_supervised_learning/data/raw"


def load_triplet_from_metadata(row):
    dapi_path = os.path.join(BASE_DIR, row["Image_PathName_DAPI"], row["Image_FileName_DAPI"]),
    tubulin_path = os.path.join(BASE_DIR, row["Image_PathName_Tubulin"], row["Image_FileName_Tubulin"])
    actin_path = os.path.join(BASE_DIR, row["Image_PathName_Actin"], row["Image_FileName_Actin"])

    dapi = tiff.imread(dapi_path)
    tubulin = tiff.imread(tubulin_path)
    actin = tiff.imread(actin_path)

    return np.stack([dapi, tubulin, actin], axis=0)  # (3, H, W)


# In[6]:


row = metadata_with_moa.iloc[0]
img = load_triplet_from_metadata(row)

print(img.shape)  # should be (3, 1024, 1280)


# In[7]:


print("Shape:", img.shape)
print("Min:", img.min())
print("Max:", img.max())


# 6 Intensity Clipping

# In[8]:


def clip_intensity(img):

    clipped = np.zeros_like(img)

    for c in range(img.shape[0]):

        low = np.percentile(img[c], 0.01)
        high = np.percentile(img[c], 99.9)

        clipped[c] = np.clip(img[c], low, high)

    return clipped


# In[9]:


img_clipped = clip_intensity(img)

print("Before max:", img.max())
print("After max:", img_clipped.max())


# In[10]:


plt.hist(img[0].flatten(), bins=100)
plt.title("Before clipping")
plt.show()

plt.hist(img_clipped[0].flatten(), bins=100)
plt.title("After clipping")
plt.show()


# In[13]:


print("Clipped min:", img_clipped.min())
print("Clipped max:", img_clipped.max())

for c in range(3):
    print(f"Channel {c} min:", img_clipped[c].min())
    print(f"Channel {c} max:", img_clipped[c].max())


# 7 Scale Intensities to [0,1]

# In[14]:


def scale_to_unit(img):

    scaled = np.zeros(img.shape, dtype=np.float32)

    for c in range(img.shape[0]):

        min_val = img[c].min()
        max_val = img[c].max()

        scaled[c] = (img[c] - min_val) / (max_val - min_val + 1e-8)

    return scaled


# In[17]:


img_scaled = scale_to_unit(img_clipped)

print("Scaled min:", img_scaled.min())
print("Scaled max:", img_scaled.max())
print("dtype:", img_scaled.dtype)


# 8 Compute Otsu DNA mask

# In[20]:


def compute_dna_mask(img_scaled):

    dna = img_scaled[0]

    thresh = threshold_otsu(dna)

    mask = dna > thresh

    return mask, thresh


# In[21]:


mask, thresh = compute_dna_mask(img_scaled)

print("Otsu threshold:", thresh)

plt.imshow(mask, cmap="gray")
plt.title("DNA mask")
plt.show()


# white regions = nuclei

# 9 Random crop sampler (224×224)

# In[22]:


def random_crop(img, mask, crop_size=224):

    C, H, W = img.shape

    y = np.random.randint(0, H - crop_size)
    x = np.random.randint(0, W - crop_size)

    crop_img = img[:, y:y+crop_size, x:x+crop_size]
    crop_mask = mask[y:y+crop_size, x:x+crop_size]

    return crop_img, crop_mask


# 10 Keep crops with cells (>=1% DNA)

# In[23]:


def sample_valid_crop(img_scaled, mask, crop_size=224):

    for _ in range(50):

        crop_img, crop_mask = random_crop(img_scaled, mask, crop_size)

        dna_fraction = crop_mask.mean()

        if dna_fraction >= 0.01:
            return crop_img

    return None


# In[24]:


crop = sample_valid_crop(img_scaled, mask)

print("Crop shape:", crop.shape)


# 11 Visualize crop

# In[25]:


fig, axes = plt.subplots(1,3, figsize=(10,4))

for i in range(3):
    axes[i].imshow(crop[i], cmap="gray")
    axes[i].axis("off")

plt.show()


# 12 Resize to training resolution

# In[26]:


def resize_for_model(img):

    img_small = resize(img, (3,128,128), anti_aliasing=True)

    return img_small


# 13 Full preprocessing pipeline

# In[27]:


def preprocess_image(row):

    img = load_triplet_from_metadata(row)

    img = clip_intensity(img)

    img = scale_to_unit(img)

    mask, _ = compute_dna_mask(img)

    crop = sample_valid_crop(img, mask)

    crop = resize(crop, (3,128,128), anti_aliasing=True)

    return crop


# 14 Full Sanity Check

# In[28]:


sample = preprocess_image(metadata_with_moa.iloc[0])

print("Final shape:", sample.shape)
print("Min:", sample.min())
print("Max:", sample.max())


# 15 Visualize final input to model

# In[29]:


fig, axes = plt.subplots(1,3, figsize=(10,4))

for i in range(3):
    axes[i].imshow(sample[i], cmap="gray")
    axes[i].axis("off")

plt.show()

