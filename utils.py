import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage import data
from skimage.io import imread,imshow
from tifffile import imread as tifread
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, disk, dilation, remove_small_objects,binary_closing
from skimage.feature import register_translation
from scipy import ndimage as ndi
from skimage.filters import sobel
import cv2

import lapjv # it is a c lib

cmap = cm.get_cmap('Paired')
X = np.linspace(0,1,12)
#X[0], X[11] = X[11], X[0]
colors = [cmap(x) for x in X]
colors.insert(0,(0,0,0,1))

selem = disk(3)
selem2 = disk(12)

# %%

def link_twoframes(image1, image2,regions1,regions2,motherLabel):
    if not regions2:
        print('mask is all black')
        regions2 = regions1
        image2 = image1

    numbCell1 = len(regions1)
    numbCell2 = len(regions2)

    C = construct_cost_matrix(regions1, regions2, image1, image2)
    aa, bb, _ = lapjv.lapjv(C)
    image2b = relabel(image2,regions1, regions2, bb, numbCell1, numbCell2)
    shift = shiftImage_distMap(image1,image2)
    mask = image2b==motherLabel
    mask1 = image1==motherLabel
    if not mask.any() and mask1.any():
        print("mother not found, try shift image")
        image2b = np.roll(image2, np.int8(shift[0]), axis=0)
        regions2b = regionprops(image2b)
        C = construct_cost_matrix(regions1, regions2b, image1, image2b)
        aa, bb, _ = lapjv.lapjv(C)
        image2b = relabel(image2b,regions1, regions2b, bb, numbCell1, numbCell2)
        image2b = np.roll(image2b, -np.int8(shift[0]),axis=0)
        mask = image2b==motherLabel
    if not mask.any() and mask1.any():
        print("mother still not found, try lower weight for smaller obj")
        C = construct_cost_matrix(regions1, regions2, image1, image2, area_weight=0)
        aa, bb, _ = lapjv.lapjv(C)
        image2b = relabel(image2,regions1, regions2, bb, numbCell1, numbCell2)
        mask = image2b==motherLabel
    prop = regionprops(mask1*image1)
    if mask.any() and len(prop)>1: 
        if abs(shift[0])>20 and prop[0].centroid[0]<85:
            print("shift too high")
            image2b = np.roll(image2, np.int8(shift[0]), axis=0)
            regions2b = regionprops(image2b)
            C = construct_cost_matrix(regions1, regions2b, image1, image2b)
            aa, bb, _ = lapjv.lapjv(C)
            image2b = relabel(image2b,regions1, regions2b, bb, numbCell1, numbCell2)
            image2b = np.roll(image2b, -np.int8(shift[0]),axis=0)
            mask = image2b==motherLabel
    motherFind = mask.any()

    return image2b,C,aa,bb,motherFind
def construct_cost_matrix(regions1, regions2, image1, image2,area_weight=75):
    #cost_link = construct_cost_link(regions1, regions2)
    cost_link = construct_cost_link2(image1, image2, regions1, regions2,area_weight)
    cost_notfind = construct_cost_notfind(regions1,image2)
    cost_newlyfound = construct_cost_newlyfound(regions2,image1, image2)
    cost_others = np.transpose(cost_link)
    cost1 = np.hstack((cost_link,cost_notfind))
    cost2 = np.hstack((cost_newlyfound,cost_others))
    cost = np.vstack((cost1, cost2))
    return cost

def construct_cost_link2(image1, image2, regions1, regions2, aweight):
    numbCell1 = len(regions1)
    numbCell2 = len(regions2)
    cost_link = np.ones([numbCell1,numbCell2])
    #ds = calculate_similarity(image1,image2,regions1,regions2)
    for ii, props1 in  enumerate(regions1):
        area1 = props1.area
        mask1 = image1==props1.label
        for jj, props2 in enumerate(regions2):
            area2 = props2.area
            mask2 = image2==props2.label
            overlap_area = sum(sum(mask1&mask2))
            if overlap_area==0:
                overlap_area=1
            ratio1 = area1/overlap_area
            ratio2 = area2/overlap_area
            if area2*1.5<area1:
                area_weight = aweight
            else:
                area_weight = 0
            cost_link[ii,jj] = ratio1 + ratio2 + area_weight 
    return cost_link
def construct_cost_notfind(regions1, image2, threshold = 0.5, coeff = 2, maxcost=100, mincost = 2):
    numbCell1 = len(regions1)
    mask2 = image2>0
    cost_notfind = np.ones([numbCell1,numbCell1])*maxcost
    for ii, props1 in  enumerate(regions1):
        area = props1.area
        area2 = sum(mask2[props1.coords[:,0], props1.coords[:,1]])
        ratio = area2/area
        cost_notfind[ii,ii] = (maxcost-mincost)*(ratio**coeff/(ratio**coeff+threshold**coeff)) + mincost
    return cost_notfind

def construct_cost_newlyfound(regions2, image1, image2, threshold = 0.5, coeff = 2, maxcost=100, mincost = 2):
    numbCell2 = len(regions2)
    mask1 = image1>0
    cost_newlyfound = np.ones([numbCell2,numbCell2])*maxcost
    for ii, props2 in  enumerate(regions2):
        area = props2.area
        overlap = (image2==props2.label) & mask1 
        area2 = sum(sum(overlap))
        ratio = area2/area
        cost_newlyfound[ii,ii] = (maxcost-mincost)*(ratio**coeff/(ratio**coeff+threshold**coeff)) + mincost
    return cost_newlyfound

# %%
def relabel(image2, regions1, regions2, bb, numbCell1, numbCell2):
    assert(len(bb)==numbCell1+numbCell2, 'bb length is not right')
    all_labels_image1 = set([props.label for props in regions1])
    available_labels = set(range(1,13)) - all_labels_image1
    image_tmp2 = np.zeros(image2.shape,dtype='uint8')
    for ii in range(numbCell2):
        mask = image2==regions2[ii].label
        if bb[ii]<numbCell1:
            # relabel second frame
            image_tmp2[mask] = regions1[bb[ii]].label
        else:
            # relabel new cells
            try:
                # this is a compromise solution, some time the algrithm gives too
                # many new cells, available_labels are not enough
                image_tmp2[mask] = available_labels.pop()
            except:
                break
    return image_tmp2

def label2rgb(label):
    # python label2rgb does not support one label on specific color, 
    # the color will be choose next even label is skipping
    rgbimage = np.zeros([label.shape[0],label.shape[1],3])
    for ii in range(label.shape[0]):
        for jj in range(label.shape[1]):
            color = colors[label[ii,jj]]
            rgbimage[ii,jj,:] = np.asarray(color[0:3])
    return rgbimage

def shiftImage_dilateCentroid(image1, image2, regions1, regions2):
    image1_tmp = np.zeros(image1.shape, dtype=np.uint16)
    image2_tmp = np.zeros(image2.shape, dtype=np.uint16)
    for props in regions1:
        centroid = np.uint8(props.centroid)
        image1_tmp[centroid[0], centroid[1]] = 1
        image1_tmp = dilation(image1_tmp, selem)
    for props in regions2:
        centroid = np.uint8(props.centroid)
        image2_tmp[centroid[0], centroid[1]] = 1
        image2_tmp = dilation(image2_tmp, selem)
    shifts,_,_ = register_translation(image1_tmp, image2_tmp) 
    return shifts

def shiftImage_distMap(image1, image2):
    distance1 = ndi.distance_transform_edt(image1)
    distance2 = ndi.distance_transform_edt(image2)
    shifts, _, _ =register_translation(distance1, distance2)
    return shifts

def shiftImage_edge(image1, image2):
    edge1 = sobel(image1>0)
    edge1 = dilation(edge1, selem)
    edge2 = sobel(image2>0)
    edge2 = dilation(edge2, selem)
    shifts, _, _ =register_translation(edge1,edge2 )
    return shifts

def shiftImage_wholeMask(image1, image2):
    shifts,_,_ = register_translation(image1>0,image2>0)
    return shifts

def calculate_similarity(image1, image2, regions1,regions2):
    ds = np.zeros([len(regions1), len(regions2)])
    for jj, props1 in enumerate(regions1):
        obj_image1 = np.uint8(props1.filled_image)
        for kk, props2 in enumerate(regions2):
            obj_image2 = np.uint8(props2.filled_image)
            d = cv2.matchShapes(obj_image1,obj_image2,cv2.CONTOURS_MATCH_I2,0)
            ds[jj,kk] = d
    return ds
def removeOverlapSmallObj(image):
    regions = regionprops(image)
    regions2 = []
    for props1 in regions:
        mask = image==props1.label
        mask = binary_closing(mask, selem2)*props1.label
        regions2.append(regionprops(mask)[0])

    for props1 in regions2:
        for props2 in regions2:
            x1 = props1.centroid[1]
            x2 = props2.centroid[1]
            y1 = props1.centroid[0]
            y2 = props2.centroid[0]
            r1 = props1.equivalent_diameter/2
            r2 = props2.equivalent_diameter/2
            dist = sqrt((x1-x2)**2+(y1-y2)**2)
            if r1>r2 and dist<r1:
                print("remove small overlaping")
                label2change = props2.label
                labelchange2 = props1.label
                image[image==label2change] = labelchange2
    return image



# %%
