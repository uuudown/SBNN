## @file process_imagenet_file.py
## @to change the path for imagenet_files.txt from imagenet_file.csv  
##
## @author Ang Li (PNNL)



import string
import os


fin = open("imagenet_files.csv","r")
fout = open("imagenet_files.txt","w")

for line in fin.readlines():
    if line.startswith("./Datasets"):
        header = "/home/lian599/raid/data/imagenet/train"
        s = line[line.rfind("/"):]
        fout.write(str(header) + s)
fin.close()
fout.close()
