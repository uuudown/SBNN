## @file process_one_image.py
## @to ready one image by PyTorch. This is used to verify whether the image reading function works properly.
##
## @author Ang Li (PNNL)

from PIL import Image
import torchvision.transforms as TF

loader = TF.Compose([
         TF.Scale(256), #The same as TF.Resize(256,Image.BILINEAR),
         TF.transforms.CenterCrop(224),
         TF.ToTensor(),
         TF.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224,0.225])
         ])

image = Image.open('./2900.JPEG')
image = loader(image).float()

print image.shape
print image[0][10]
