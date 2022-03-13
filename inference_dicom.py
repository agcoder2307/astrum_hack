""" Inferencing Dicom volume first, and then send it to the PACS

1. Identify the series to run MaskRCNN algorithm on from a folder containing multiple studies
2. Construct a numpy volume from a set of dicom files
3. Run inference on the constructed volume
4. Call a shell script to push report to the storage archive


"""

import os
import sys
import time
from IPython.display import clear_output
import shutil
import pydicom

# Model stuff
from inference.mrcnn import model as modellib
from inference import prediction_conf
from inference.mrcnn.model import mold_image
from numpy import expand_dims
import cv2 as cv

# Session and security
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from dicom_modify import dicom_to_numpy
from report import save_report_as_dcm, create_report
from itertools import count

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
cfg = prediction_conf.PredictionConfig()
sess = tf.compat.v1.Session(config=config)
set_session(sess)

model = modellib.MaskRCNN(mode='inference', model_dir='inference/', config=cfg)
model.load_weights('inference/mask_rcnn_tumor_cfg_0300.h5', by_name=True)
graph = tf.compat.v1.get_default_graph()


def load_dicom_volume_as_numpy_from_list(dcmlist):
    """Loads a list of PyDicom objects a numpy array.

    Assumes that only one series is in the array

    Arguments:
        dcmlist -- path to directory

    Returns:
        tuple of (3d volume, header of the 1st image)

    """

    images_per_view = [dicom_to_numpy(dcm) for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)]

    header_for_images = []

    for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber):
        dcm.PixelData = None
        header_for_images.append(dcm)
    # Make sure that you have correctly constructed the volume from your axial slices!

    # We return header so that we can inspect metadata properly.
    # Since for our purposes we are interested in "Series" header, we grab header of the
    # first file (assuming that any instance-specific values will be ighored - common approach)
    # We also zero-out Pixel Data since the users of this function are only interested in metadata

    return (images_per_view, header_for_images)


def predict(raw_images):
    """Predicts images using model

    Arguments:
        raw_images {list} -- list of numpy array images

    Returns:
        new_images {list} -- list of predicted images
    """
    new_images = []
    for i, image in enumerate(raw_images):
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        with graph.as_default():
            set_session(sess)
            pred = model.detect(sample, verbose=0)[0]
        for t, box in enumerate(pred['rois']):
            y1, x1, y2, x2 = box
            if pred["class_ids"][t] == 1:
                label = '2'
                color = (0,0,255)
            elif pred["class_ids"][t] == 2:
                label = '4'
                color = (255,0,0)
            elif pred["class_ids"][t] == 3:
                label = "4a"
                color = (0,255,0)
            elif pred["class_ids"][t] == 4:
                label = "4b"
                color = (0,255,255)
            elif pred["class_ids"][t] == 5:
                label = "4c"
                color = (255,0,255)
            else:
                label = '5'
                color = (255,255,0)
            cv.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv.putText(image, label+ " - " + "{:.2f}".format(pred["scores"][t]), (x1, y1-40), cv.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        new_images.append(image)

    return new_images
    

def get_series_for_inference(path):
    """Reads multiple series from one folder and picks the one to run
    inference on.

    Arguments:
        path {string} -- location of the DICOM files

    Returns:
        Numpy array representing the series
    """

    # Path is a directory that contains a full study as a collection of files
    # We are reading all files into a list of PyDciom objects so that we can filter them later
    dicoms = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path)]

    # create a series_for_inference containing a list of only those Python objects that represents files
    # Label your dicom files sent by scanner if you need 
    
    series_for_inference = []
    
    # only accept mammogram for inference
    [series_for_inference.append(d) for d in dicoms if d.Modality == "MG"]
    
    # if len({f.SeriesInstanceUID for f in series_for_inference}) != 1:
    #     print("Error: cannot figure out what series to run inference on")
    #     return []

    return series_for_inference


def os_command(command):
    # Comment this if running under Windows
    # sp = subprocess.Popen(["/bin/bash", "-i", "c", command])
    # sp.communicate()

    os.system(command)


def delete_last_line():
    "Use this function to delete the last line in the STDOUT"

    #cursor up one line
    sys.stdout.write('\x1b[1A')

    #delete last line
    sys.stdout.write('\x1b[2K')


if __name__ == "__main__":

    # This code expects a single command line argument with link to the directory containing
    # routed studies

    if len(sys.argv) != 2:
        print("You should supply one command line argument pointing to the routing folder. Exiting.")
        sys.exit()

    # Find all subdirectories within the supplied directory. We assume that
    # one subdirectory contains a full study
    condition = True
    while condition:

        subdirs = [os.path.join(sys.argv[1], d) for d in os.listdir(sys.argv[1]) if
               os.path.isdir(os.path.join(sys.argv[1], d))]

        print(subdirs)

        if subdirs:
            # Get the latest directory
            print(subdirs)
            time.sleep(10)
            study_dir = sorted(subdirs, key=lambda dir: os.stat(dir).st_mtime, reverse=True)[0]

            print(study_dir)
            print(f"Looking for instances to run inference on in directory {study_dir}...")

            images_per_view, header_for_images = load_dicom_volume_as_numpy_from_list(get_series_for_inference(study_dir))

            if len(images_per_view) == 4:
                print(f"Found {len(images_per_view)} instances")

                print("AIMED: Running inference....")

                inferenced_images = predict(images_per_view)

                print("Creating and pushing the report to the AIMED PACS")
                # for number, (image, header) in enumerate(zip(inferenced_images, header_for_images)): not memory efficient

                for number, image, header in zip(count(), inferenced_images, header_for_images):
                    # report_img = create_report(header, image)
                    save_report_as_dcm(header, image, f"./dicomDiagnosed/report{number}.dcm")

                    # sending the dcm to the PACS
                    os_command(f'storescu localhost 4242 -v -aec BREASTCANCERAI +r +sd -xs ./dicomDiagnosed/report{number}.dcm')

                print('Images are saved and sent to the pacs')

                # deleting the dcm files inside the dicomDiagnosed folder
                # for number, numpy_array in enumerate(inferenced_images):
                #     os.remove(f"./dicomDiagnosed/report{number}.dcm")

                print('Sucessfully deleted all the files in the dicomDiagnosed directory!')

                # deleting the study dir
                time.sleep(2)
                shutil.rmtree(study_dir, onerror=lambda f, p, e: print(f"Error deleting: {e[1]}"))
            else:
                print(f"Found {len(images_per_view)} instances")
                print('Looking for 4 images')

        else:
            clear_output(wait=True)
            print("No directory")
            print("Waiting for a study...")
            time.sleep(10)
            continue

