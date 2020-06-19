import cv2
import os
import json
import numpy as np
import editdistance
from spellchecker import SpellChecker
from linesegm2.LineSegmentation import LineSegmentation
from Model import Model
from SamplePreprocessor import preprocess
from DataLoader import Batch
from WordSegmentation import wordSegmentation, prepareImg

def infer(model, fnImg):
	"recognize text in image provided by file path"
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	spell = SpellChecker() 

	# find those words that may be misspelled 
	#misspelled = spell.unknown([recognized]) 

	recognized[0] = spell.correction(recognized[0])
	#print('Recognized:', '"' + recognized[0] + '"')
	#print('Probability:', probability[0])
	return (recognized[0], probability[0])


def line_segment(filepath, filenames, model):
	out_dict={}
	out_path='../data/out/'
	truth_path='../data/true_text/'
	compare=False

	if os.path.exists(truth_path+'truth.json'):
		numCharErr = 0
		numCharTotal = 0
		numWordOK = 0
		numWordTotal = 0
		compare = True
		with open(truth_path+'truth.json', 'r') as truth:
			truth_file = json.load(truth)

	for filename in filenames:
		fullpath=os.path.join(filepath,filename)
		f=filename.split('.')[0]
		ext=filename.split('.')[1]
		if ext=='pdf':
			continue
		out_dict[f] = []
		print('Reading image "' + filename + '"..')
		im = cv2.imread(fullpath)

		output_path = out_path+f
		if not os.path.exists(output_path):
			os.mkdir(output_path)
		line_segmentation = LineSegmentation(img=im, output_path=output_path)
		lines = line_segmentation.segment()

		if(len(lines)==0):
			im = cv2.imread(fullpath, 0)
			_ , imbw = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
			lines = [imbw]
		n_line=1
		n_word=0

		for line in lines:
			img = prepareImg(line, 50)
			# execute segmentation with given parameters
			# -kernelSize: size of filter kernel (odd integer)
			# -sigma: standard deviation of Gaussian function used for filter kernel
			# -theta: approximated width/height ratio of words, filter function is distorted by this factor
			# - minArea: ignore word candidates smaller than specified area
			res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100, increase_dim=10)
			
			# iterate over all segmented words
			#print('Segmented into %d words'%len(res))
			for (j, w) in enumerate(res):
				(wordBox, wordImg) = w
				(x, y, w, h) = wordBox
				imgloc=output_path+'/%d.png'%j
				# increase contrast
				# preprocess so that it is similar to IAM dataset
				kernel = np.ones((2, 2), np.uint8)
				wordImg = cv2.erode(wordImg, kernel, iterations = 1)
				cv2.imwrite(imgloc, wordImg) # save word
				#FilePaths.fnInfer = 'out/%s/%d.png'%(f,j)
				#result, prob = infer(model, imgloc)
				try:
					result, prob = infer(model, imgloc)
				except:
					print("Couldn't infer: image%d"%j)
					result=""
				#compare with ground truth
				if compare:
					numWordOK += 1 if truth_file[f][n_word] == result else 0
					numWordTotal += 1
					dist = editdistance.eval(result, truth_file[f][n_word])
					numCharErr += dist
					numCharTotal += len(truth_file[f][n_word])
					print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + truth_file[f][n_word] + '"', '->', '"' + result + '"')
				#updating output dictionary
				out_dict[f].append(result)
				n_word+=1
				#deleting intermediate file
				os.remove(imgloc)
				cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
			
			# output summary image with bounding boxes around words
			cv2.imwrite(output_path+'/summary%d.png'%n_line, img)
			n_line+=1

	if compare:
		charErrorRate = numCharErr / numCharTotal
		wordAccuracy = numWordOK / numWordTotal
		print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
	return out_dict