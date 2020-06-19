import cv2
import os
import numpy as np
from sys import argv
from lib import sauvola, linelocalization, pathfinder
from WordSegmentation import wordSegmentation, prepareImg
from time import time as timer
from SamplePreprocessor import preprocess
from DataLoader import Batch
from Model import Model


def draw_line(im, path):
	for p in path:
		im[p[0], p[1]] = 0

def draw_box(im, path, prev):
	curr=(path[0][1], path[0][0])
	cv2.rectangle(im, prev, curr,0,3)


def draw_map(im, mapy):
	for m in mapy:
		im[m[0], m[1]] = 255


def print_path(path):
	print('\t# path: ' + str(path[::-1]))


def save(filename, imbw, immap):
	imbw_filename = str.replace(filename, '.', '_bw.')
	#imbw_filename = str.replace(imbw_filename, 'data', 'data/bw')
	print('Saving image "' + imbw_filename + '"..\n')
	cv2.imwrite(imbw_filename, imbw)
	immap_filename = str.replace(imbw_filename, '_bw', '_map')
	cv2.imwrite(immap_filename, immap)

def infer(model, fnImg):
	"recognize text in image provided by file path"
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	#print('Recognized:', '"' + recognized[0] + '"')
	#print('Probability:', probability[0])
	return (recognized[0], probability[0])

######################
# ------ MAIN ------ #
######################

def identify_words(filepath, filenames, model):
	begin = timer()
	out_dict={}
	out_path='../data/out/'
	

	print('############################')
	print('#Line and Word Segmentation#')
	print('############################')

	for filename in filenames:
		fullpath=os.path.join(filepath,filename)
		f=filename.split('.')[0]
		ext=filename.split('.')[1]
		if ext=='pdf':
			continue
		out_dict[f] = []
		print('Reading image "' + filename + '"..')
		im = cv2.imread(fullpath, 0)

		print('- Thresholding image..')
		#imbw = sauvola.binarize(im, [20, 20], 128, 0.3)
		pxmin = np.min(im)
		pxmax = np.max(im)
		im = (im - pxmin) / (pxmax - pxmin) * 255
		im = im.astype('uint8')
		#binarize
		_ , imbw = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		#cv2.imshow('tempb', imbw)

		# increase line width
		kernel = np.ones((3, 3), np.uint8)
		imbw = cv2.erode(imbw, kernel, iterations = 1)

		print('- Localizing lines..')
		lines = linelocalization.localize(imbw)
		lines.append(imbw.shape[0])
		print(' => ' + str(len(lines)) + ' lines detected.')

		print('- Path planning with ')

		immap = np.zeros((imbw.shape), dtype=np.int32)
		# for i in range(0, 1):
		prev=(0,0)
		n_line=1
		for line in lines:
			# line = lines[i]
			rline = imbw[int(prev[1]):int(line),:]
			img = prepareImg(rline, 50)
			# execute segmentation with given parameters
			# -kernelSize: size of filter kernel (odd integer)
			# -sigma: standard deviation of Gaussian function used for filter kernel
			# -theta: approximated width/height ratio of words, filter function is distorted by this factor
			# - minArea: ignore word candidates smaller than specified area
			res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100, increase_dim=10)
			
			# write output to 'out/inputFileName' directory
			if not os.path.exists(out_path+'%s'%f):
				os.mkdir(out_path+'%s'%f)
			
			# iterate over all segmented words
			#print('Segmented into %d words'%len(res))
			for (j, w) in enumerate(res):
				(wordBox, wordImg) = w
				(x, y, w, h) = wordBox
				imgloc=out_path+'%s/%d.png'%(f, j)
				# increase contrast
				cv2.imwrite(imgloc, wordImg) # save word
				#FilePaths.fnInfer = 'out/%s/%d.png'%(f,j)
				try:
					result, prob = infer(model, imgloc)
				except:
					print("Couldn't infer: image%d"%j)
					result=""
				#updating output dictionary
				out_dict[f].append(result)
				#deleting intermediate file
				##os.remove(imgloc)
				cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
			
			# output summary image with bounding boxes around words
			cv2.imwrite(out_path+'%s/summary%d.png'%(f,n_line), img)

			#path, mapy = pathfinder.search(imbw, 'A', line)
			#path = [[int(line),0]]
			path = [[int(line),rline.shape[1]]]
			#print(rline.shape[1])
			#print('path[0][0]: ',path[0][0], ' path[0][1]: ', path[0][1])
			draw_box(im, path, prev)
			#draw_map(immap, mapy)
			prev=(0, path[0][0])
			n_line+=1
			# print_path(path)

		#save(filename, imbw, immap)
		cv2.imwrite(out_path+'%s/summary.png'%f, im)
	return out_dict

	print(' - Elapsed time: ' + str((timer() - begin)) + ' s')