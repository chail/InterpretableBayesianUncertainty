import os
import numpy as np

def parse_classes(filehandle, outputdir):

    classes = []
    for lineno, line in enumerate(filehandle):
        if lineno == 0:
            class_names = line.strip().split(',')[1:]
            [classes.append([]) for c in class_names]
        else:
            data = line.strip().split(',')
            im = data[0]
            classno = np.argmax(data[1:])
            classes[classno].append(im)

    for (img_list, name) in zip(classes, class_names):
        with open(os.path.join(outputdir, name + '.txt'), 'w') as f:
            [f.write('{}\n'.format(im)) for im in img_list]

if __name__ == '__main__':
    labelfile = os.path.join('ISIC2018_Task3_Training_GroundTruth',
                             'ISIC2018_Task3_Training_GroundTruth.csv')
    classes_dir = 'classes'
    os.makedirs(classes_dir, exist_ok=True)

    with open(labelfile, 'r') as f:
        parse_classes(f, classes_dir)
