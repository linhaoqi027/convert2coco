import cv2
import numpy as np
import os, glob
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
import os
import re
import datetime
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask
from tqdm import tqdm

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name.split('/')[-1],
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info

def create_annotation_info(annotation_id, image_id, category_info, binary_mask,
                           image_size=None, tolerance=2, bounding_box=None):

    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)

    # print(binary_mask.shape)
    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    if category_info["is_crowd"]:
        is_crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else :
        is_crowd = 0
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)
        if not segmentation:
            return None

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }

    return annotation_info




ROOT_DIR = '/data/cityscapes/val'
IMAGE_DIR = os.path.join(ROOT_DIR, "images/frankfurt")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "gt/frankfurt")
INSTANCE_DIR = os.path.join(ROOT_DIR, "instances")

INFO = {
    "description": "Smoke_Instance Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": "2021",
    "contributor": "Haoqi_Lin",
    "date_created": "2021-5-03 16:16:16.123456"
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'smoke',
        'supercategory': 'smoke_coco',
    },
]


#
# def masks_generator(imges):
#     idx = 0
#     for pic_name in imges:
#         annotation_name = pic_name.split('_')[0] + '_' + pic_name.split('_')[1] + '_' + pic_name.split('_')[
#             2] + '_gtFine_instanceIds.png'
#         print(annotation_name)
#         annotation = cv2.imread(os.path.join(ANNOTATION_DIR, annotation_name), -1)
#         name = pic_name.split('.')[0]
#         h, w = annotation.shape[:2]
#         ids = np.unique(annotation)
#         for id in ids:
#             if id in background_label:
#                 continue
#             instance_id = id
#             class_id = instance_id // 1000
#             if class_id == 24:
#                 instance_class = 'pedestrian'
#             elif class_id == 25:
#                 instance_class = 'rider'
#             elif class_id == 26:
#                 instance_class = 'car'
#             elif class_id == 27:
#                 instance_class = 'truck'
#             elif class_id == 28:
#                 instance_class = 'bus'
#             else:
#                 continue
#             print(instance_id)
#             instance_mask = np.zeros((h, w, 3), dtype=np.uint8)
#             mask = annotation == instance_id
#             instance_mask[mask] = 255
#             mask_name = name + '_' + instance_class + '_' + str(idx) + '.png'
#             cv2.imwrite(os.path.join(INSTANCE_DIR, mask_name), instance_mask)
#             idx += 1


# def filter_for_pic(files):
#     file_types = ['*.jpeg', '*.jpg', '*.png']
#     file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
#     files = [f for f in files if re.match(file_types, f)]
#     # files = [os.path.join(root, f) for f in files]
#     return files
#
#
# def filter_for_instances(root, files, image_filename):
#     file_types = ['*.png']
#     file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
#     files = [f for f in files if re.match(file_types, f)]
#     basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
#     file_name_prefix = basename_no_extension + '.*'
#     # files = [os.path.join(root, f) for f in files]
#     files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
#     return files


def main():
    # for root, _, files in os.walk(ANNOTATION_DIR):
    # files = os.listdir(IMAGE_DIR)
    # image_files = filter_for_pic(files)
    # masks_generator(image_files)

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    image_id = 1
    segmentation_id = 1

    # basepath = '/home/ecust/lhq/instance-seg-smoke/data/composite_methane_video_v2'
    # rgb_train = np.loadtxt(os.path.join(basepath, 'metadata', 'testing_image_paths' + '.txt'), dtype='str')

    # basepath = '/home/ecust/blender_dataset/composite_gas_2_segmentation_10000_500x400'
    basepath='/home/ecust/lhq/real_gas_test'
    rgb_train = np.loadtxt(os.path.join(basepath, 'test_data_real' + '.txt'), dtype='str')

    # go through each image
    for image_path in tqdm(rgb_train):
        image = Image.open(image_path)
        image = image.resize((512, 512))
        image_info = create_image_info(
            image_id, image_path, image.size)
        coco_output["images"].append(image_info)

        # filter for associated png annotations
        # for root, _, files in os.walk(INSTANCE_DIR):
        annotation_files = image_path.replace('pic', 'mask')
        annotation = cv2.imread(annotation_files, 0)
        annotation=cv2.resize(annotation, (512,512), interpolation=cv2.INTER_NEAREST)
        #annotation=np.ones((512,512))


        # annotation = Image.open(annotation_files).convert('L')
        # annotation = annotation.resize((512, 512), Image.NEAREST)
        ids = np.unique(annotation)
        h, w = 512,512
        for id in ids:
            if id ==0:
                continue
            instance_id=id
            instance_mask = np.zeros((h, w), dtype=np.uint8)
            mask = annotation == instance_id
            instance_mask[mask] = 255
            category_info = {'id': 1,'is_crowd': 0}
            annotation_info = create_annotation_info(
                segmentation_id, image_id, category_info, instance_mask,
                (512,512), tolerance=2)
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
            segmentation_id = segmentation_id + 1
        image_id = image_id + 1


    with open('instances_test2017_2.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

if __name__ == "__main__":
    main()

