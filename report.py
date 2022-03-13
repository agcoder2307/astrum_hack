from PIL import ImageFont
from PIL import ImageDraw
import datetime
import numpy as np
import pydicom
from PIL import Image
import SimpleITK as sitk


def create_report(header, predicted_numpy):
    image = Image.new("RGB", (4728, 5928))
    draw = ImageDraw.Draw(image)

    header_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=40)
    main_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=20)

    draw.text((50, 50), "AI RESULT", (255, 255, 255), font=header_font)

    draw.multiline_text((50, 140),
                        f"Patient ID: {header.PatientID} \n \
                        Study Description : {header.StudyDescription}\n \
                        Modality: {header.Modality}\n \
                        Image Type: {header.ImageType}\n",
                        (255, 255, 255), font=main_font)

    # original = np.flip((predicted_numpy[0, :, :]/np.max(predicted_numpy[0, :, :]))*0xff).T.astype(np.uint8)

    # pil_orig = Image.fromarray(original, mode="L").convert("RGBA").resize((400, 400))

    # image.paste(pil_orig, box=(4728, 5928))

    return image


def save_report_as_dcm(header, predicted_array, path):
    out = pydicom.Dataset(header)

    out.file_meta = pydicom.Dataset()
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian    #ExplicitVRLittleEndian

    out.is_little_endian = True
    out.is_implicit_VR = False

    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"

    out.file_meta.MediaStorageSOPClassUID = out.SOPClassUID

    out.SeriesInstanceUID = pydicom.uid.generate_uid()
    out.SOPInstanceUID = pydicom.uid.generate_uid()

    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.Modality = "MG"
    out.SeriesDescription = "AI RESULT"

    rows, columns, channel = predicted_array.shape
    out.Rows = rows
    out.Columns = columns

    out.ImageType = r"DERIVED\PRIMARY\AXIAL"  # deriving this image from patient data
    out.SamplesPerPixel = 3  # building an RGB image.
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0  # bytes encode pixels as R1G1B1R2G2B2... as opposed to R1R2R3...G1G2G3...
    out.BitsAllocated = 8  # using 8 bits/pixel
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0

    dt = datetime.date.today().strftime("%Y%m%d")
    tm = datetime.datetime.now().strftime("%H%M%S")
    out.StudyDate = dt
    out.StudyTime = tm
    out.SeriesDate = dt
    out.SeriesTime = tm
  
    out.ImagesInAcquisition = 1

    # empty these since most viewers will then default to auto W/L
    out.WindowCenter = ""
    out.WindowWidth = ""

    # Data imprinted directly into image pixels is called "burned in annotation"
    out.BurnedInAnnotation = "YES"

    #out.PixelData = pydicom.encaps.encapsulate([predicted_array.tobytes()])
    # out['PixelData'].is_undefined_length = True
    out.PixelData = predicted_array.tobytes()
    
    pydicom.filewriter.dcmwrite(path, out, write_like_original=False)

"../dicomDiagnosed"
