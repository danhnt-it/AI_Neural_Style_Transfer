# USAGE
# python F:/HK4/AI/Project_Style_Transfer/neural-style-transfer/neural_style_transfer.py --image F:/HK4/AI/Project_Style_Transfer/neural-style-transfer/images/baden_baden.jpg --model F:/HK4/AI/Project_Style_Transfer/neural-style-transfer/models/instance_norm/starry_night.t7
# python F:/HK4/AI/Project_Style_Transfer/neural-style-transfer/neural_style_transfer.py --image F:/HK4/AI/Project_Style_Transfer/neural-style-transfer/images/UIT_E_building.jpg --model F:/HK4/AI/Project_Style_Transfer/neural-style-transfer/models/instance_norm/the_scream.t7

# import the libraries
import argparse
import imutils
import time
import cv2
import tkinter as tk
from tkinter import filedialog
import random

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="neural style transfer model")
# ap.add_argument("-i", "--image", required=True,
# 	help="input image to apply neural style transfer to")
# args = vars(ap.parse_args())

root = tk.Tk()
root.title("Artificial Intelligence v0.0")
root.geometry("400x300")
root.configure(background='black')

def close_window():
    root.destroy()

def choose_content():
    root.contentname = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("model files","*.t7"))) #("all files","*.*")
    print(root.contentname)

def choose_style():
    root.stylename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("model files","*.t7"),("jpeg files","*.jpg")))
    print(root.stylename)

def transfer():
	# load the neural style transfer model from disk
	print("[INFO] loading style transfer model...")
	net = cv2.dnn.readNetFromTorch(root.stylename) # load a pre-trained neural style transfer model into memory as net

	# load the input image, resize it to have a width of 600 pixels, and 
	# then grab the image dimensions
	image = cv2.imread(root.contentname) # load the input image via argparse
	image = imutils.resize(image, width=600) # resize input image
	(h, w) = image.shape[:2] # image.shape[0] & image.shape[1]

	# construct a blob from the image, set the input, and then perform a forward pass of the network
	# facilitate image processing for deep learning classification
	blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), 
		(103.939, 116.779, 123.680), swapRB=False, crop=False) 
	net.setInput(blob) # set a new input value for the network
	start = time.time() # for counting purpose
	output = net.forward() # run forward pass for the whole network to compute the output of layers
	end = time.time()

	# post-process the output image
	# reshape the output tensor, add back in the mean subtraction, and
	# then swap the channel ordering
	output = output.reshape((3, output.shape[2], output.shape[3])) # reshape the matrix to simply be (3, H, W)
	# add back in the mean values subtracted previously
	output[0] += 103.939
	output[1] += 116.779
	output[2] += 123.680
	output /= 255.0 #scaling 
	output = output.transpose(1, 2, 0) # transpose the matrix to channels-last ordering

	# show information on how long inference took
	print("[INFO] neural style transfer took {:.4f} seconds".format(end - start))

	numrand = random.randint(1,99999)
	# show the images
	cv2.imshow("Input", image)
	cv2.imshow("Output", output)
	output_cv = output * 255
	output_cv = output_cv.astype('uint8')
	cv2.imwrite("F:/HK4/AI/Project_Style_Transfer/neural-style-transfer/Output/result%d.jpg" % numrand, output_cv)
	# cv2.waitKey(0)

lbl = tk.Label(root, text = "\nNEURAL STYLE TRANSFER\n",bg = "black", fg = "white", font = ('Helvetica', 18))
lbl.pack(side = "top")

btn_choosefile = tk.Button(root, text = "Choose Content", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 15), command = choose_content)
btn_choosefile.pack(side = "top")

btn_choosefile = tk.Button(root, text = "Choose Style", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 15), command = choose_style)
btn_choosefile.pack(side = "top")

btn_choosefile = tk.Button(root, text = "Start Transfer", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 15), command = transfer)
btn_choosefile.pack(side = "top")

btn = tk.Button(root, text = "Close", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 15), command = close_window)
btn.pack(side = "top")

root.mainloop()