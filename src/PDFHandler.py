import os
from wand.image import Image as wi

def convertPDF(pdfs):
	##Creates images 
	if len(pdfs)==0:
		return
	for file in pdfs:
		pdf = wi(filename=file, resolution=300)
		pdfimage = pdf.convert("png")
		i=1
		file=file[:-4]
		for img in pdfimage.sequence:
			page = wi(image=img)
			page.save(filename=file+'_'+str(i)+".png")
			i +=1

def deleteTemp(pdfs):
	if len(pdfs)==0:
		return
	for file in pdfs:
		file=file[:-4]
		i=1
		while(True):
			imgpath=file+'_'+str(i)+".png"
			if os.path.exists(imgpath):
				os.remove(imgpath)
			else:
				break
			i+=1