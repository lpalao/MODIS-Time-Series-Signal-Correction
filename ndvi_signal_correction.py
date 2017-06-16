# MODIS-Time-Series-Signal-Correction
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 03 14:49:48 2015

@author: LPalao
script for drought assessment and monitoring
Use linear interpolation to correct for cloud pixels
"""
import gdal, gdalnumeric, osr
from gdalconst import *
from PyQt4.QtGui import QInputDialog
import numpy as np
import os, glob, itertools, time, gc

start = time.ctime()

############ Function definition

def array_stack(ndvi,qflag,xrows1,xrows2,yrows1,yrows2): #this should be list
    index=np.arange(0,len(ndvi))
    arrstack=np.empty(shape=(46,abs(xrows2-xrows1),abs(yrows2-yrows1))) #dtype=float
    for nv,q,x in itertools.izip(ndvi,qflag,index):     
        arrndvi=gdalnumeric.LoadFile(nv)[xrows1:xrows2,yrows1:yrows2]  
        arrqflag=gdalnumeric.LoadFile(q)[xrows1:xrows2,yrows1:yrows2] 
        mask_clouds=np.where(arrqflag==3,np.nan,arrndvi)
        arrstack[x,:,:]=mask_clouds
    return arrstack

def interpolate(array):
    arrN=np.array(array, copy=False)
    ix=-np.isnan(arrN)
    idxTrue=ix.nonzero()[0]
    valueTrue=arrN[-np.isnan(arrN)]
    countFalse=np.isnan(arrN).ravel().nonzero()[0]
    arrN[np.isnan(arrN)]=np.interp(countFalse,idxTrue,valueTrue)
    #print arrN
    #return arrN
    
def array_to_raster(array,filename):
    dst_filename = filename
    proj=ds.GetProjection()
    x_pixels = ds.RasterXSize  # number of pixels in x
    y_pixels = ds.RasterYSize  # number of pixels in y
    PIXEL_SIZE = dsTrans[1]  # size of the pixel...        
    x_min = dsTrans[0] 
    y_max = dsTrans[3]  # x_min & y_max are like the "top left" corner.
    wkt_projection = proj
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(dst_filename,x_pixels,y_pixels,1,gdal.GDT_Int16, ) # GDT_Float32
    dataset.SetGeoTransform((x_min,PIXEL_SIZE,0,y_max,0,-PIXEL_SIZE))  
    dataset.SetProjection(wkt_projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.GetRasterBand(1).SetNoDataValue(-32768) # added
    dataset.FlushCache()
    print os.path.basename(filename)
    return dataset

def arrMerge(directory):
    os.chdir(directory)
    
    files=yfndvi[index]    
    
    for i in range(len(yfndvi[index])):
        fileout=out_interp + "\\" + files[i].rsplit(".",-1)[0][:25] + "NDVI_Interpol" + ".tif"
        
        #IB=Interpolated Block
        col1=np.vstack([np.load("IB0\\IB%03d.npy" % i),np.load("IB5\\IB%03d.npy" % i),np.load("IB10\\IB%03d.npy" % i),np.load("IB15\\IB%03d.npy" % i),np.load("IB20\\IB%03d.npy" % i)]) 
        col2=np.vstack([np.load("IB1\\IB%03d.npy" % i),np.load("IB6\\IB%03d.npy" % i),np.load("IB11\\IB%03d.npy" % i),np.load("IB16\\IB%03d.npy" % i),np.load("IB21\\IB%03d.npy" % i)])
        col3=np.vstack([np.load("IB2\\IB%03d.npy" % i),np.load("IB7\\IB%03d.npy" % i),np.load("IB12\\IB%03d.npy" % i),np.load("IB17\\IB%03d.npy" % i),np.load("IB22\\IB%03d.npy" % i)])
        col4=np.vstack([np.load("IB3\\IB%03d.npy" % i),np.load("IB8\\IB%03d.npy" % i),np.load("IB13\\IB%03d.npy" % i),np.load("IB18\\IB%03d.npy" % i),np.load("IB23\\IB%03d.npy" % i)])
        col5=np.vstack([np.load("IB4\\IB%03d.npy" % i),np.load("IB9\\IB%03d.npy" % i),np.load("IB14\\IB%03d.npy" % i),np.load("IB19\\IB%03d.npy" % i),np.load("IB24\\IB%03d.npy" % i)])
        
        x=np.hstack([col1,col2,col3,col4,col5])
        array_to_raster(x,fileout)
        del col1,col2,col3,col4,col5
    
    for root, dirs, files in os.walk(out_npy):
        for basename in files:
            os.remove(os.path.join(root, basename))
    
    gc.collect()

###################

#directory=os.path.abspath("J:\Test\Extract_SubD")
directory=QInputDialog.getText(None,"Folder of TIFF Files", "Please specify folder path")[0]
os.chdir(directory)

fndvi=glob.glob("*NDVI.tif")
fqflag=glob.glob("*PIXR.tif")

ds=gdal.Open(fndvi[0], GA_ReadOnly)
dsTrans=ds.GetGeoTransform()

driver=ds.GetDriver()

yfndvi=[]
yqflag=[]
        
for k,v in itertools.groupby(fndvi,key=lambda x:x[4:10]):
    yfndvi.append(list(v))

for k,v in itertools.groupby(fqflag,key=lambda x:x[4:10]):
    yqflag.append(list(v))

ws=os.getcwd()
out_npy = os.path.split(ws)[0] + "\\" + "Blocks" # Output extract band
if not os.path.exists(out_npy):
	os.makedirs(out_npy)

out_interp = directory
if not os.path.exists(out_interp):
	os.makedirs(out_interp)

x_total,y_total=ds.RasterXSize,ds.RasterYSize #rastersizeXY
n_chunks=6 #800x800, if 9 600x600 if 5 1200x1200

#prepare the chunk indices
x_offsets=np.linspace(0,x_total,n_chunks).astype(int)
x_offsets=zip(x_offsets[:-1],x_offsets[1:])
y_offsets=np.linspace(0,y_total,n_chunks).astype(int)
y_offsets=zip(y_offsets[:-1],y_offsets[1:])

index=0
while index < len(yfndvi):
    os.chdir(directory)
    step=0
    
    for x1,x2 in x_offsets:
        for y1,y2 in y_offsets:
            #arrstack=np.empty(shape=(46,1200,1200))
            arrVeg=array_stack(yfndvi[index],yqflag[index],x1,x2,y1,y2)
        
            dim=arrVeg.shape   
            for r in range(0,dim[1]):
                for c in range(0,dim[2]): 
                    if np.count_nonzero(np.isnan(arrVeg[:,r,c])) == 46:
                        print "row: %s, col: %s" %(r,c)
                        arrVeg[:,r,c][np.isnan(arrVeg[:,r,c])]=0
                    else:
                        interpolate(arrVeg[:,r,c])
                        #test.append(arrVeg[:,r,c])
                    
            #arrVeg.astype(int)
            
            out_tab=out_npy + "\\IB%s" %step
            if not os.path.exists(out_tab):
                os.makedirs(out_tab)
            
            for r in range(dim[0]):   
                tab_save=np.save(out_tab + "\\" + "IB%03d.npy" %r, arrVeg[r,:,:])
            step += 1 
    
    arrMerge(out_npy)
   
    end = time.ctime()
    print 'processing start time:%s, end time:%s' %(start,end)    

    index += 1
    gc.collect()
