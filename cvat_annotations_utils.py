import xmltodict
import string
import numpy as np
import functools
import mask2polygons

class Cvat_annotation:
    def __init__(self, cvat_xml):

        self.images_annotation: list[Image_annotation] = self.read_images_xml(cvat_xml['annotations'].get('image', []))

    def read_images_xml(self, images_xml):
        images_annotation = []
        for image_xml in images_xml if isinstance(images_xml, list) else [images_xml]:
            images_annotation.append(Image_annotation(image_xml))
        return images_annotation
    
    def convert_cvat_to_yolo(self):

        bboxes_unique_labels: list[str] = []
        masks_unique_labels: list[str] = []
        

        images_with_bbox_filenames: list[str] = []
        images_with_mask_filenames: list[str] = []

        images_with_bbox_annotations_yolo: list[str] = []
        images_with_mask_annotations_yolo: list[str] = []

        # saber os labels ids de cada label: saber quais ids sao de bbox e quais de mask
        for image_annotation in self.images_annotation:
            
            bboxes_yolo_string: list[str] = []
            masks_yolo_string: list[str] = []
            
            image_has_bbox = False
            image_has_mask = False
            
            for box_annotation in image_annotation.boxes_annotation:
                
                image_has_bbox = True
                
                # Save unique existing labels, to later use as yolo class id
                if box_annotation.label not in bboxes_unique_labels:
                    bboxes_unique_labels.append(box_annotation.label)
                

                # Convert cvat corner coords to yolo center coords, width and height
                bbox_x_center = (box_annotation.xtl + box_annotation.xbr) / 2
                bbox_y_center = (box_annotation.ytl + box_annotation.ybr) / 2
                bbox_width = box_annotation.xbr - box_annotation.xtl
                bbox_height = box_annotation.ybr - box_annotation.ytl
                # Normalize yolo coords witht the full image size, as per yolo standard
                bbox_x_center /= image_annotation.width
                bbox_y_center /= image_annotation.height
                bbox_width /= image_annotation.width
                bbox_height /= image_annotation.height


                bboxes_yolo_string.append(f"{box_annotation.label} {bbox_x_center} {bbox_y_center} {bbox_width} {bbox_height}")

            for mask_annotation in image_annotation.masks_annotation:

                image_has_mask = True    

                # Save unique existing labels, to later use as yolo class id
                if mask_annotation.label not in masks_unique_labels:
                    masks_unique_labels.append(mask_annotation.label)

                # Convert cvat rle to binary mask and then to polygon
                mask_image = rle_to_binary_image_mask(mask_annotation.rle, mask_annotation.top, mask_annotation.left, mask_annotation.width, image_annotation.height, image_annotation.width)
                mask_polygons = mask2polygons.mask2polygon(mask_image)

                # I only labeled one mask per mask image, but if there were more this will add each as a seperate line in the yolo format
                for mask_polygon in mask_polygons:
                    # Normalize polygon x and y coords, as per yolo standard
                    for i in range(len(mask_polygon)):
                        if i % 2 == 0:            
                            mask_polygon[i] /= image_annotation.width # Even index, x coord, remembering contour was fallten in mask2polygon
                        else:               
                            mask_polygon[i] /= image_annotation.height # Odd index, y coord

                    mask_polygon_str = " ".join(str(coord) for coord in mask_polygon)
                    masks_yolo_string.append(f"{mask_annotation.label} {mask_polygon_str}")



            # Only store filename if, in fact, it has annotations compatibale with yolo (bbox or mask)
            if(image_has_bbox==True):
                images_with_bbox_filenames.append(image_annotation.name)
                images_with_bbox_annotations_yolo.append("\n".join(bboxes_yolo_string))
                
            if(image_has_mask==True):
                images_with_mask_filenames.append(image_annotation.name)
                images_with_mask_annotations_yolo.append("\n".join(masks_yolo_string))

        # Reorder the list alphabetically, since yolo seems to do this for ids
        bboxes_unique_labels = sorted(bboxes_unique_labels)
        masks_unique_labels = sorted(masks_unique_labels)

         
        # Replace the yolo class label with the id, 
        # Id is the index of the alphabetically ordered unique existing labels
        # Note: not creating a function for this cuz i dont feel like deepcopying and changing by inference is kinda not clear
        # Note: Also theres surelly a more efficient way to replace this, since im reading the whole string multiple times to replace, but right now this is fine
        for i in range(len(images_with_bbox_annotations_yolo)):
            for j in range(len(bboxes_unique_labels)):
                images_with_bbox_annotations_yolo[i] = images_with_bbox_annotations_yolo[i].replace(bboxes_unique_labels[j], str(j))

        for i in range(len(images_with_mask_annotations_yolo)):
            image_with_mask_annotation_yolo = images_with_mask_annotations_yolo[i]
            for j in range(len(masks_unique_labels)):
                images_with_mask_annotations_yolo[i] = images_with_mask_annotations_yolo[i].replace(masks_unique_labels[j], str(j))
        
        assert len(images_with_bbox_filenames) == len(images_with_bbox_annotations_yolo)
        assert len(images_with_mask_filenames) == len(images_with_mask_annotations_yolo)

        return images_with_bbox_filenames, images_with_bbox_annotations_yolo, bboxes_unique_labels, images_with_mask_filenames, images_with_mask_annotations_yolo, masks_unique_labels


class Image_annotation:
    def __init__(self, image_xml):

        self.id = int(image_xml['@id'])
        self.name = str(image_xml['@name'])
        self.width = int(image_xml['@width'])
        self.height = int(image_xml['@height'])
        self.boxes_annotation: list[Box_annotation] = self.read_boxes_xml(image_xml.get('box', []))
        self.all_points_annotation: list[Points_annotation] = self.read_points_xml(image_xml.get('points', []))
        self.masks_annotation: list[Mask_annotation] = self.read_masks_xml(image_xml.get('mask', []))

    def read_boxes_xml(self, boxes_xml):
        boxes_annotation = []
        for box_xml in boxes_xml if isinstance(boxes_xml, list) else [boxes_xml]:
            boxes_annotation.append(Box_annotation(box_xml))
        return boxes_annotation
    
    def read_points_xml(self, all_points_xml):
        points_annotation = []
        for points_xml in all_points_xml if isinstance(all_points_xml, list) else [all_points_xml]:
            points_annotation.append(Points_annotation(points_xml))
        return points_annotation
    
    def read_masks_xml(self, masks_xml):
        masks_annotation = []
        for mask_xml in masks_xml if isinstance(masks_xml, list) else [masks_xml]:
            masks_annotation.append(Mask_annotation(mask_xml))
        return masks_annotation
    
    def get_all_bboxes(self):
        all_bboxes = []
        for box_annotation in self.boxes_annotation:
            all_bboxes.append(box_annotation.bbox)
        return all_bboxes
    
    def get_all_points_coords(self):
        all_points_coords = []
        for points_annotation in self.all_points_annotation:
            all_points_coords.append(points_annotation.points_coords)
        return all_points_coords

class Box_annotation:
    def __init__(self, box_xml):

        self.label = str(box_xml['@label'])
        self.source = str(box_xml['@source'])
        self.occluded = str(box_xml['@occluded'])
        self.xtl = float(box_xml['@xtl'])
        self.ytl = float(box_xml['@ytl'])
        self.xbr = float(box_xml['@xbr'])
        self.ybr = float(box_xml['@ybr'])
        self.z_order = int(box_xml['@z_order'])

class Points_annotation:
    def __init__(self, points_xml):

        self.label = str(points_xml['@label'])
        self.source = str(points_xml['@source'])
        self.occluded = str(points_xml['@occluded'])
        self.points_coords = self.parse_points_str(points_xml['@points'])
        self.z_order =  int(points_xml['@z_order'])


    def parse_points_str(self, points_str):
        # points_coords = [(x0,y0), (x1,y1), ...]
        points_coords = []

        points_seperated_str = points_str.split(';') # points="x0,y0;x1,y1;"
        for point in points_seperated_str:
            x, y = map(float, point.split(','))
            points_coords.append((x,y))

        return points_coords

class Mask_annotation:
    def __init__(self, mask_xml):

        self.label = str(mask_xml['@label'])
        self.source = str(mask_xml['@source'])
        self.occluded = str(mask_xml['@occluded'])
        self.rle = self.parse_rle_str(mask_xml['@rle'])
        self.left = int(mask_xml['@left'])
        self.top = int(mask_xml['@top'])
        self.width = int(mask_xml['@width'])
        self.height = int(mask_xml['@height'])
        self.z_order =  int(mask_xml['@z_order'])

    def parse_rle_str(self, rle_str):
        return list(map(int, rle_str.split(', '))) # Deserialzie rle values into list, values are separated by ", "



def read_xml_file(xml_filepath):
    try:
        with open(xml_filepath, 'r') as xml_file:
            xml_string = xml_file.read()
            xml_dict = xmltodict.parse(xml_string)
            return xml_dict
    except (FileNotFoundError, OSError) as e:
        print(f"Exception in read_xml_file(): '{xml_filepath}', {e}")
        raise # Re-raise original exeception

def rle_to_binary_image_mask(rle, mask_top, mask_left, mask_width, image_height, image_width):
    # https://github.com/opencv/cvat/issues/6487
    # zhiltsov-max commented Jul 18, 2023

    mask_image = np.zeros((image_height, image_width), dtype=np.uint8)
    value = 0
    offset = 0
    for rle_count in rle:
        while rle_count > 0:
            y, x = divmod(offset, mask_width)
            mask_image[y + mask_top][x + mask_left] = value
            rle_count -= 1
            offset += 1
        value = 1 - value

    return mask_image

def binary_image_mask_to_cvat_mask_rle(mask_image):
    # https://github.com/opencv/cvat/issues/6487
    # zhiltsov-max commented Jul 18, 2023

    # Determine bbox of mask
    istrue = np.argwhere(mask_image == 1).transpose()
    top = int(istrue[0].min())
    left = int(istrue[1].min())
    bottom = int(istrue[0].max())
    right = int(istrue[1].max())
    roi_mask = mask_image[top:bottom + 1, left:right + 1]

    # Compute RLE values
    def reduce_fn(acc, v):
        if v == acc['val']:
            acc['res'][-1] += 1
        else:
            acc['val'] = v
            acc['res'].append(1)
        return acc
    
    roi_rle = functools.reduce(
        reduce_fn,
        roi_mask.flat,
        { 'res': [0], 'val': False }
    )['res']

    width = right - left + 1
    height = bottom - top + 1

    return roi_rle, top, left, width, height