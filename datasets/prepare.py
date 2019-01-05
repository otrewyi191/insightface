import os

imgdir = '/home/zzx/github.com/otrewyi191/MSCELEB1M-GenImage/low_images'

listdir = os.listdir(imgdir)

with open('/home/zzx/github.com/otrewyi191/insightface/datasets/train.lst','w') as lst:

    count = 0
    for face_dir in listdir:
        # each man
        img_fullpath = os.path.join(imgdir, face_dir)

        img_list = os.listdir(img_fullpath)
        for img in img_list:
            path_join = os.path.join(img_fullpath, img)
            lst.write("1\t{path}\t{count}\n".format(path=path_join,count = count))

        count = count + 1
