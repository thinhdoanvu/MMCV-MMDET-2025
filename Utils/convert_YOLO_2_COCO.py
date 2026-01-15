import logging

logging.getLogger().setLevel(logging.CRITICAL)

from pylabel import importer

# path_to_annotations = r"C:\Users\VU\Documents\OBD\AICUP25\labels\train"
path_to_annotations =r"C:\Users\VU\Documents\OBD\AICUP25\labels\val"

# Identify the path to get from the annotations to the images
# path_to_images = r"C:\Users\VU\Documents\OBD\AICUP25\images\train"
path_to_images = r"C:\Users\VU\Documents\OBD\AICUP25\images\val"

# Import the dataset into the pylable schema
# Class names are defined here https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml
yoloclasses = ['aortic_valve']
# dataset = importer.ImportYoloV5(path=path_to_annotations, path_to_images=path_to_images, cat_names=yoloclasses, img_ext="png", name="instances_train2017")
dataset = importer.ImportYoloV5(path=path_to_annotations, path_to_images=path_to_images, cat_names=yoloclasses, img_ext="png", name="instances_val2017")

dataset.df.head(5)
print(f"Number of images: {dataset.analyze.num_images}")
print(f"Number of classes: {dataset.analyze.num_classes}")
print(f"Classes:{dataset.analyze.classes}")
print(f"Class counts:\n{dataset.analyze.class_counts}")

dataset.export.ExportToCoco(cat_id_index=1)
