from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
#import numpy as np

app = Flask(__name__)

dic = {0: 'OK', 1: 'NG'}

model = load_model('model.h5')

model.make_predict_function()


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(64, 64))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 64, 64, 3)
    pred_score = model.predict(i)
    p = (model.predict(i) > 0.5).astype("int32")
    pred_score *= 100

    #p = model.predict_image(img_path,'model.h5')
    if p==0:
        a = 100-pred_score[0][0]
        result = 'NG'
    else:
        a = pred_score[0][0]
        result = 'OK'
    return [result,round(a,2)]


# def predict_image(img_path, model):
#     predict = image.load_img(img_path, target_size=(64, 64))
#     predict_modified = image.img_to_array(predict)
#     predict_modified = predict_modified / 255
#     predict_modified = np.expand_dims(predict_modified, axis=0)
#     result = model.predict(predict_modified)
#     if result[0][0] >= 0.5:
#         prediction = 'OK'
#         probability = result[0][0]
#         probability *= 100
#         probability = round(probability, 2)
#
#     else:
#         prediction = 'NG'
#         probability = 1 - result[0][0]
#         probability *= 100
#         probability = round(probability, 2)
#     return prediction


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "Please subscribe  Artificial Intelligence Hub..!!!"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        #img_path = img.filename
        img.save(img_path)

        p = predict_label(img_path)

    return render_template("index.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    # app.debug = True
    app.run(debug=True)
