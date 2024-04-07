from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)



model = load_model('model.h5')

# model.make_predict_function()

def predict_label(img_path):
	i=image.load_img(img_path, target_size=(224, 224))
	i=image.img_to_array(i)
	i=np.expand_dims(i, axis=0)
	i = i/255.0
	out=model.predict(i)
	f=''
	pred_labels=np.argmax(out, axis=1)
	idx_to_classes={3:'healthy',2:'fussarium_wilt',1:'curl_virus',0:'bacterial_blight'}
	for key in idx_to_classes.keys():
		if pred_labels[0]==key:
			f=idx_to_classes[key]
			break
	return f




# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)