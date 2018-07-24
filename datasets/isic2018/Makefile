
all: ISIC2018_Task3_Training_Input resize classes 

ISIC2018_Task3_Training_Input:
	unzip isic2018_task3_images.zip

ISIC2018_Task3_Training_GroundTruth:
	unzip isic2018_task3_labels.zip

classes:
	python extract_classes.py

resize: ISIC2018_Task3_Training_Input
	mkdir -p $@
	bash resize_image.sh $< $@

