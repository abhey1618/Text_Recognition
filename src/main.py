from __future__ import division
from __future__ import print_function

import sys
import os
import json
from glob import glob as glob
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
from WordSegmentation import wordSegmentation, prepareImg
from PDFHandler import convertPDF, deleteTemp
from linesegm import identify_words
from linesegm_new import line_segment


class FilePaths:
	"filenames and paths to data"
	fnCharList = '../model/charList.txt'
	fnAccuracy = '../model/accuracy.txt'
	fnTrain = '../data/'
	fnInfer = '../data/test.png'
	fnCorpus = '../data/corpus.txt'
	fnLineData = '../data/lines/'


def train(model, loader):
	"train NN"
	epoch = 0 # number of training epochs since start
	bestCharErrorRate = float('inf') # best valdiation character error rate
	noImprovementSince = 0 # number of epochs no improvement of character error rate occured
	earlyStopping = 5 # stop training after this number of epochs without improvement
	while True:
		epoch += 1
		print('Epoch:', epoch)

		# train
		print('Train NN')
		loader.trainSet()
		while loader.hasNext():
			iterInfo = loader.getIteratorInfo()
			batch = loader.getNext()
			loss = model.trainBatch(batch)
			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

		# validate
		charErrorRate = validate(model, loader)
		
		# if best validation accuracy so far, save model parameters
		if charErrorRate < bestCharErrorRate:
			print('Character error rate improved, save model')
			bestCharErrorRate = charErrorRate
			noImprovementSince = 0
			model.save()
			open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
		else:
			print('Character error rate not improved')
			noImprovementSince += 1

		# stop training if no more improvement in the last x epochs
		if noImprovementSince >= earlyStopping:
			print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
			break


def validate(model, loader):
	"validate NN"
	print('Validate NN')
	loader.validationSet()
	numCharErr = 0
	numCharTotal = 0
	numWordOK = 0
	numWordTotal = 0
	while loader.hasNext():
		iterInfo = loader.getIteratorInfo()
		print('Batch:', iterInfo[0],'/', iterInfo[1])
		batch = loader.getNext()
		(recognized, _) = model.inferBatch(batch)
		
		print('Ground truth -> Recognized')	
		for i in range(len(recognized)):
			numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			dist = editdistance.eval(recognized[i], batch.gtTexts[i])
			numCharErr += dist
			numCharTotal += len(batch.gtTexts[i])
			print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
	
	# print validation result
	charErrorRate = numCharErr / numCharTotal
	wordAccuracy = numWordOK / numWordTotal
	print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
	return charErrorRate

def infer(model, fnImg):
	"recognize text in image provided by file path"
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	print('Recognized:', '"' + recognized[0] + '"')
	print('Probability:', probability[0])
	return (recognized[0], probability[0])


def LineData(decoderType,dump):
	"""reads images from data/ and outputs the word-segmentation to out/"""

	# read input images from 'in' directory
	imgFiles = os.listdir(FilePaths.fnLineData)
	if not os.path.exists('../data/out/'):
		os.mkdir('../data/out')
	print(open(FilePaths.fnAccuracy).read())
	out_dict={}
	model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=dump)

	for (i,f) in enumerate(imgFiles):
		#print('Segmenting words of sample %s'%f)
		file_compo=f.split('.')
		fname=f
		f=file_compo[0]
		extension=file_compo[1]
		out_dict[f] = []
		
		if extension == 'pdf':
			continue

		# read image, prepare it by resizing it to fixed height and converting it to grayscale
		img = prepareImg(cv2.imread(FilePaths.fnLineData+fname), 50)
		
		# execute segmentation with given parameters
		# -kernelSize: size of filter kernel (odd integer)
		# -sigma: standard deviation of Gaussian function used for filter kernel
		# -theta: approximated width/height ratio of words, filter function is distorted by this factor
		# - minArea: ignore word candidates smaller than specified area
		res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
		
		# write output to 'out/inputFileName' directory
		if not os.path.exists('../data/out/%s'%f):
			os.mkdir('../data/out/%s'%f)
		
		# iterate over all segmented words
		#print('Segmented into %d words'%len(res))
		for (j, w) in enumerate(res):
			(wordBox, wordImg) = w
			(x, y, w, h) = wordBox
			cv2.imwrite('../data/out/%s/%d.png'%(f, j), wordImg) # save word
			FilePaths.fnInfer = '../data/out/%s/%d.png'%(f,j)
			result, prob = infer(model, FilePaths.fnInfer)
			#updating output dictionary
			out_dict[f].append(result)
			#deleting intermediate file
			os.remove('../data/out/%s/%d.png'%(f, j))
			cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
		
		# output summary image with bounding boxes around words
		cv2.imwrite('../data/out/%s/summary.png'%f, img)
		#generating json output
		json_object=json.dumps(out_dict, indent=4)
		with open('../data/out/output.json', "w") as outfile:
			outfile.write(json_object)

def main():
	"main function"
	# optional command line args
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train the NN', action='store_true')
	parser.add_argument('--validate', help='validate the NN', action='store_true')
	parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
	parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding', action='store_true')
	parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')
	parser.add_argument('--line', help='the image is of a line', action='store_true')
	parser.add_argument('--doc', help='the image is of multiple lines', action='store_true')

	args = parser.parse_args()
	new_segm=True

	decoderType = DecoderType.BestPath
	if args.beamsearch:
		decoderType = DecoderType.BeamSearch
	elif args.wordbeamsearch:
		decoderType = DecoderType.WordBeamSearch

	# train or validate on IAM dataset	
	if args.train or args.validate:
		# load training data, create TF model
		loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

		# save characters of model for inference mode
		open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
		
		# save words contained in dataset into file
		open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

		# execute training or validation
		if args.train:
			model = Model(loader.charList, decoderType)
			train(model, loader)
		elif args.validate:
			model = Model(loader.charList, decoderType, mustRestore=True)
			validate(model, loader)

	# infer text on test image
	elif args.line:
		pdfs = glob(os.path.join(FilePaths.fnLineData, '*.pdf'))
		convertPDF(pdfs)
		LineData(decoderType,args.dump)
		deleteTemp(pdfs)

	elif args.doc:
		pdfs = glob(os.path.join(FilePaths.fnLineData, '*.pdf'))
		convertPDF(pdfs)
		imgFiles = os.listdir(FilePaths.fnLineData)
		if not os.path.exists('../data/out/'):
			os.mkdir('../data/out')
		print(open(FilePaths.fnAccuracy).read())
		model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)

		if new_segm:
			out_dict = line_segment(FilePaths.fnLineData, imgFiles, model)
		else:
			out_dict = identify_words(FilePaths.fnLineData, imgFiles, model)
		json_object=json.dumps(out_dict, indent=4)
		with open('../data/out/output.json', "w") as outfile:
			outfile.write(json_object)
		deleteTemp(pdfs)

	else:
		print(open(FilePaths.fnAccuracy).read())
		model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
		_,_=infer(model, FilePaths.fnInfer)


if __name__ == '__main__':
	main()
