import urllib
import requests
import os
from BeautifulSoup import BeautifulSoup
from PIL import Image
import shutil

numberOfImages = 150
categories = ["bar","botanical_garden","windmill","street"]

for category in categories:
	staticURL = "http://labelme.csail.mit.edu/Release3.0/Images/users/antonio/static_sun_database/" + category[0] + "/" + category + "/"
	categoryURL = "http://groups.csail.mit.edu/vision/SUN/scenes/pages/" + category[0] + "/" + category + "/"
	if not os.path.exists(category): os.makedirs(category)
	response = requests.get(categoryURL+"index.html")
	soup = BeautifulSoup(response.text)
	images = soup.findAll("div",{"class":"nonannoimage-box"})
	document = open(category + '/images.txt', 'w')
	count = 0
	for image in images:
		if count > numberOfImages:
			break
		imageName = image.text.split("src")[1].split('"')[1]
		if imageName == " target=":
			continue
		print imageName + " (" + str(count) + " of " + str(numberOfImages) + ")"
		urllib.urlretrieve(staticURL + imageName, category + "/" + imageName)
		im = Image.open(category + "/" + imageName)
		im.load()
		if im.size[0] > 500 or im.size[1] > 500:
			basewidth = 500
			img = Image.open(category + "/" + imageName)
			wpercent = (basewidth/float(img.size[0]))
			hsize = int((float(img.size[1])*float(wpercent)))
			img = img.resize((basewidth,hsize), Image.ANTIALIAS)
			img.save(category + "/" + imageName)
		count += 1
		document.write(imageName + '\n')
	document.close()