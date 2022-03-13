import pydicom
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import cv2 as cv2


# def numpy_reshape(np_array):

    # cv2.imwrite('images/image.png', np_array)

    # np_array = cv2.imread('images/image.png')

    #np_array = cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB)

    #return np_array

def dicom_to_numpy(dicom, voi_lut = True, fix_monochrome = True):

    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               

    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    # Reshaping the numpy array to (5928, 4728, 3)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    return data


