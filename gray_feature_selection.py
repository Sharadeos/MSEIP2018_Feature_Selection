from time import sleep
from PIL import Image
import glob
import numpy as np


#takes all the images in the current directory and loads them into a list.
filelist = glob.glob('image.jpg')
print(filelist)
#please change to the appropriate directory

#takes the list of images and converts it into a 4d array x[0][0][0][0]
x = np.array([np.array(Image.open(fname)) for fname in filelist])


#shows the first image's specications
jpgfile = Image.open(filelist[0])
#jpgfile.show(command='fim')
print(jpgfile.bits, jpgfile.size, jpgfile.format)

counter = 0
list(jpgfile.getdata())
for pixel in iter(jpgfile.getdata()):
    #print(pixel)
    counter+=1

print (counter, "pixels for this image")

print(x[0][0][0], " the first pixel in the first image RGB")
print(len(x[0]), " Length")
print(len(x[0][0]), " Width")



#converts the x 4d array into a 2d array with flattened array.

print("The 1st image, with a flattened array for all the rgb values")
final = x.flatten().reshape(len(x),len(x[0])*len(x[0][0])*len(x[0][0][0]))


sample_gray = [[0 for x in range(220)] for z in range(220)]

#print(len(sample_gray))
#print(len(sample_gray[0]))


for x1 in range(len(sample_gray)):
    for y1 in range(len(sample_gray[0])):
       sample_gray[x1][y1] += ((int(x[0][x1][y1][0])*0.3) + (int(x[0][x1][y1][1])*0.6) +(int(x[0][x1][y1][2])*0.1)) 

img = Image.fromarray(x[0], 'RGB')
img_array= np.array(Image.open(filelist[0]).convert('L'))


#convert to numpy array

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

imageCount = 1
imageSize = 220
columnSize = 220

sample_gray = np.array(sample_gray)

sample_gray = np.asarray(sample_gray, dtype='int32')
feature_select = np.reshape(sample_gray, (imageCount, imageSize*columnSize))



print("Feature Selection")
# Feature Selection
# 64 x 64 greyscale image


column_counter = 0
columnAccumulator = []#[[0 for x in range(imageSize*imageSize)] for y in range(imageCount)] 


columnDerivatives = [0 for x in range(columnSize)]
columnDerivativesLabel = [0 for x in range(columnSize)]

for columnCounter in range(columnSize):
	
	columnAccumulator = []
	for imageCounter in range(imageCount):
		
		
		for columnMover in range(imageSize):
		
			columnAccumulator.append(feature_select[imageCounter][(imageSize*columnMover)+columnCounter])
	#print(columnAccumulator)

	
	columnDerivatives[columnCounter] = np.std(columnAccumulator)
	print("Standard Deviation for Column:", columnCounter, "is", columnDerivatives[columnCounter])
	if columnDerivatives[columnCounter] > 5 :
		columnDerivativesLabel[columnCounter] = 1
		column_counter+=1

print(columnDerivativesLabel)

#making new images out of selected columns
#there is an easier way to this, use transpose()

feature_select_image = [[[0 for x in range(column_counter)] for z in range(220)]  for y in range(imageCount)]

print(len(feature_select_image))
print(len(feature_select_image[0]))
print(len(feature_select_image[0][0]))
offset = 0
for imageCounter in range(imageCount):

	for columnCounter in range(columnSize):	
		if columnDerivativesLabel[columnCounter] == 1:
			for columnMover in range(imageSize):
			
				feature_select_image[imageCounter][columnMover][offset] = feature_select[imageCounter][(imageSize*columnMover)+columnCounter]
			offset +=1
		
			

			
#print(feature_select_image[0])
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

feature_image = np.array(feature_select_image[0])

sample_gray = np.asarray(sample_gray, dtype='uint8')
print(sample_gray)
print(img_array)

img2 = Image.fromarray(sample_gray, 'L')

print(img2)
img2.save('image_gray.jpg')
#img2.show(command='fim')


print(final[0])
print(final[0][0]/4.3)
final.astype(float)
#1st image
final[0][0] = float(final[0][0]/4.3)
print(final[0][0])

#a = np.array([[1,2], [3,4]])
#print(np.std(a))

print(feature_image)
feature_image = np.asarray(feature_image, dtype='int32')
feature_image = np.asarray(feature_image, dtype='uint8')
img3 = Image.fromarray(feature_image, 'L')
print(img3)
img3.save('image_gray_feature.jpg')
