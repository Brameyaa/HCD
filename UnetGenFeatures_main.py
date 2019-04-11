from model import *
from data import *
from preparedataforUnet import *
import os, os.path

#create pretrained weights
##data_gen_args = dict(rotation_range=0.2,
##                    width_shift_range=0.05,
##                    height_shift_range=0.05,
##                    shear_range=0.05,
##                    zoom_range=0.05,
##                    horizontal_flip=True,
##                    fill_mode='nearest')
##myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
##
##model = unet()
##model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
##model.fit_generator(myGene,steps_per_epoch=600,epochs=1,callbacks=[model_checkpoint])
##

#get convert data to correct format for UNet 
Datasets = ['Validation_data','Training_data']#,'Test_data']
labels = ['/Invasive','/Normal'] #'/Benign','/In Situ',
msgs = []
for dd in Datasets:
    for fldr in labels:
        inpath = str(dd + fldr)
        outpath = dd + '_unet2' + fldr +'/'
        cnt = gendata(inpath,outpath,512)
        msg = 'Generated '+str(cnt)+' images in '+outpath
        msgs.append(msg)
for msg in msgs:
    print(msg)

Datasets_unet = ['Training_data_unet','Validation_data_unet']
labels = ['/Benign','/In Situ','/Invasive','/Normal']

model = unet()
model.load_weights("unet_weights.hdf5")
#for trainging and validation naming doesn't matter
for dd in Datasets_unet:
    for fldr in labels:
        inpath = str(dd + fldr)
        outpath = dd + '_feat' + fldr +'/'
        numfiles = len([name for name in os.listdir(inpath) if os.path.isfile(os.path.join(inpath, name))])
        #testGenerator(inpath, str(i)+'_')
        testGene = testGenerator(inpath,'')
        results = model.predict_generator(testGene,numfiles,verbose=1)
        saveResult(outpath,results,'')
        #print("done ",i)
        break

#keeping all the diff slices together per image
Datasets_unet = ['Test_data_unet']
labels = ['/Benign','/In Situ','/Invasive','/Normal']
for dd in Datasets_unet:
    for fldr in labels:
        inpath = str(dd + fldr)
        outpath = dd + '_feat' + fldr +'/'
        for i in range(36):
            for filename in os.listdir(inpath):
                if filename.startswith(str(i)+'_'):
                    numfiles = 48#len([name for name in os.listdir(inpath) if os.path.isfile(os.path.join(inpath, name))])
                    #testGenerator(inpath, str(i)+'_')
                    testGene = testGenerator(inpath, str(i)+'_')
                    results = model.predict_generator(testGene,numfiles,verbose=1)
                    saveResult(outpath,results,str(i)+'_')
                    #print("done ",i)
                    break
                
                

#model = unet()
#model.load_weights("unet_weights.hdf5")
#results = model.predict_generator(testGene,48,verbose=1)
#full = saveResult(save_path="playdata",filename=filename,npyfile=results)
