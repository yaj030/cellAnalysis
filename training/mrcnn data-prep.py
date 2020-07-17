#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json


# In[10]:


with open('D:\Segmentation_project\Final\MaskExp.json') as f:
    data = json.load(f)
f.close()


# In[11]:


data


# In[12]:


with open('D:/Segmentation_project/Final/MRCNN_all/train/region_data.json') as f:
    data2 = json.load(f)
f.close()


# In[9]:


data2


# In[13]:


with open('D:/Segmentation_project/Final/new_masks.json') as f:
    data3 = json.load(f)
f.close()


# In[14]:


data3


# In[109]:


region_data = []
Masks_all = []
Masks_ID = []
region_data = []
for i in data3:
    if 'Masks' in i:
        Masks_all.append( i['Masks']['cell'])
        if not i['External ID'][:-4] in Masks_ID:
            Masks_ID.append(i['External ID'][:-4])
            #print(i['Masks']['cell'])
            region_data.append(i)


# In[110]:


len(Masks_ID)


# In[111]:


import  os


# In[112]:


mask_images= []
image_path = 'D:/Segmentation_project/Final/New_test/xy01c1/indv_data/'
ids = next(os.walk(image_path))[2]
for im_id in ids:
    if im_id[:-4] in Masks_ID:
        mask_images.append(im_id)


# In[116]:


mask_images[0]


# In[92]:


import collections
print([item for item, count in collections.Counter(Masks_ID).items() if count > 1])


# In[115]:


len(mask_images)


# In[102]:


len(region_data)


# In[118]:


import cv2


# In[169]:


mask_images= []
image_path = 'D:/Segmentation_project/Final/New_test/xy01c1/indv_data/'
ids = next(os.walk(image_path))[2]
for im_id in ids:
    if im_id[:-4] in Masks_ID:
        img = cv2.imread('D:/Segmentation_project/Final/New_test/xy01c1/indv_data/'+im_id,cv2.IMREAD_ANYDEPTH)
        cv2.imwrite('D:/Segmentation_project/Final/MRCNN_all/train2/'+im_id, img)


# In[124]:


for i in region_data:
   #print(img['External ID'][0:-4] +'.tif')
    try:
        i['Label']['Cell']
        flag = 'Cell'
    except:
        flag = 'cell'
    


# In[140]:


region_data = []
region_data = data2
Masks_all = []
Masks_ID = []
for i in data3:
    if 'Masks' in i:
        Masks_all.append( i['Masks']['cell'])
        if not i['External ID'][:-4] in Masks_ID:
            Masks_ID.append(i['External ID'][:-4])
            #print(i['Masks']['cell'])
            region_data.append(i)


# In[162]:


len(region_data)


# In[143]:


1570+352


# In[152]:


a.extend(b)
a


# In[158]:


a = [1,2,3,4,5,6,7]
b= [3,4,5,6,7]


# In[159]:


c = a
for j in b:
    c.append(j)


# In[160]:


c


# In[161]:


a


# In[165]:


with open('D:/Segmentation_project/Final/MRCNN_all/train2/regioin_data.json', 'w') as outfile:  
    json.dump(region_data, outfile)


# In[166]:


with open('D:/Segmentation_project/Final/MRCNN_all/train2/region_data.json') as f:
    datat = json.load(f)
f.close()


# In[168]:


len(datat)


# In[ ]:




