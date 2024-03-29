import globox
import os
import numpy as np
import functools
import cvat_annotations_utils

cvat_filepath=""
yolo_folderpath=""

def create_yolo_annotations_folder(images_filenames, images_annotations_yolo, unique_labels, folderpath):

    obj_train_data_folderpath = os.path.join(folderpath, "obj_train_data")

    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    
    if not os.path.exists(obj_train_data_folderpath):
        os.makedirs(obj_train_data_folderpath)

    # Create .txt annotations files
    for i in range(len(images_annotations_yolo)):
         
        annotation_filename = images_filenames[i].split(".")[0] + ".txt" # Note that images_filenames is the same len as images_annotations_yolo
        annotation_filepath = os.path.join(obj_train_data_folderpath, annotation_filename)

        image_annotation = images_annotations_yolo[i]

        with open(annotation_filepath, "w") as f:
            f.write(image_annotation)


    # When exporting yolo format directly from cvat, it created these files in this structure
    # Basicaly only obj.names file is important, since it gives the annotations in the order of their ids
    # Create obj.names file
    obj_names_filepath = os.path.join(folderpath, "obj.names")
    with open(obj_names_filepath, "w") as f:
        for i in range(len(unique_labels)):
            print(f"Label ID: {i}, Label: {unique_labels[i]}")
            
            f.write(f"{unique_labels[i]}\n")

    # Create obj.data file
    obj_data_filepath = os.path.join(folderpath, "obj.data")
    with open(obj_data_filepath, "w") as f:
        f.write(f"classes = {len(unique_labels)}\n")
        f.write(f"train = data/train.txt\n")
        f.write(f"names = data/obj.names\n")
        f.write(f"backup = backup/\n")

def convert_annotations_cvat_to_yolo(cvat_filepath, yolo_annotation_folderpath):
    if not cvat_filepath.endswith('.xml'):
        print(f"Error, annoation_xml_filepath={cvat_filepath} is not .xml.")
        return

    if not os.path.exists(yolo_annotation_folderpath):
        os.makedirs(yolo_annotation_folderpath)

    cvat_xml = cvat_annotations_utils.read_xml_file(cvat_filepath)
    cvat_annotation = cvat_annotations_utils.Cvat_annotation(cvat_xml)
    images_with_bbox_filenames, images_with_bbox_annotation_yolo, bboxes_unique_labels, images_with_mask_filenames, images_with_mask_annotation_yolo, masks_unique_labels = cvat_annotation.convert_cvat_to_yolo()

    if len(images_with_bbox_annotation_yolo) != 0:

        yolo_bboxes_annotation_folderpath = os.path.join(yolo_annotation_folderpath, "bboxes_annotations")
        if not os.path.exists(yolo_bboxes_annotation_folderpath):
            os.makedirs(yolo_bboxes_annotation_folderpath)

        create_yolo_annotations_folder(images_with_bbox_filenames, images_with_bbox_annotation_yolo, bboxes_unique_labels, yolo_bboxes_annotation_folderpath)

    if len(images_with_mask_annotation_yolo) != 0:

        yolo_masks_annotation_folderpath = os.path.join(yolo_annotation_folderpath, "masks_annotations")
        if not os.path.exists(yolo_masks_annotation_folderpath):
            os.makedirs(yolo_masks_annotation_folderpath)

        create_yolo_annotations_folder(images_with_mask_filenames, images_with_mask_annotation_yolo, masks_unique_labels, yolo_masks_annotation_folderpath)

           
    print("Successfully converted CVAT to YOLO.")
    return


convert_annotations_cvat_to_yolo(cvat_filepath, yolo_folderpath)
