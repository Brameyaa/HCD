from skimage import io
from skimage import transform as trans
import numpy as np
import os
import os.path
from model import *


def gendata(inpth,opth,size):
    dir = os.path.expanduser(inpth)
    images = []
    dim = size
    total = 0
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        filename, ext = os.path.splitext(os.path.basename(d))
        filename = filename.split('_')
        #print (dir, filename[0], ext)
        img = io.imread(d,as_gray=True)
        y = []
        count = 0
        col = int(1536/dim)
        row = int(2048/dim)
        for i in range(col):
            for j in range(row):
                section = img[i * dim: (i + 1) * dim, j * dim: (j + 1) * dim]
                y.append(section)
                filepath = opth+str(filename[0])+'_'+str(count)+'.png'
                io.imsave(os.path.join(filepath),section)
                count += 1
        total += count
        #print(count,opth)
    return total
        
##def saveResul(save_path,npyfile,flag_multi_class = False,num_class = 2,picname):
##    full = np.array(1536, 2048, 1)
##    dim = 256
##    col = 0
##    row = 0
##    for i,item in enumerate(npyfile):
##        img = item[:,:,0]
##        io.imsave(os.path.join(save_path,"%d_predict.tif"%i),img)
##        full[col * dim: (col + 1) * dim, row * dim: (row + 1) * dim] = img
##        if row > (2048/dim) :
##            row = 0
##            col += 1
##        if col > (1536/dim) :
##            print('image saved')
##            break
##    io.imsave(os.path.join(save_path,"0_full.png"),full)
##    return full
##
##gendata('playdata2', 256)
##results = model.predict_generator(testGene,30,verbose=1)
##saveResult("data/membrane/test",results)

##model = unet()
##model.load_weights("unet_membrane.hdf5")
##results = model.predict_generator(testGene,(6*8),verbose=1)
##saveResult("playdata",results)
     
