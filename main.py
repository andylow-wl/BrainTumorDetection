from flask import Flask, request, render_template
import sys
import cv2
from glob import glob
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
from skimage import io


app = Flask("__name__",template_folder='template')


# Classification model
classification_model = load_model('model_classification.h5')

# Segmentation model 
def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)

segmentation_model = load_model('model_segmentation.h5',custom_objects={'dice_coef':dice_coef,'jac_distance':jac_distance,'dice_coef_loss': dice_coef_loss,"iou":iou})

classification_model.make_predict_function()

def predict_label(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img ,(256,256))
    img = img.reshape(1,256,256,3)
    img = np.array(img)
    pred1 = classification_model.predict(img)
    pred1 = np.argmax(pred1,axis=1)

    
    if pred1 == 0:
        return 'Glioma'
    elif pred1 == 1:
        return "Meningioma"
    elif pred1 == 2:
        return "No Tumour"
    
    return "Pituitary"

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


def predict_segmentation_mask(image_path):
    # reads an brain MRI image
    img = io.imread(image_path)
    img = cv2.resize(img,(256,256))
    img = np.array(img, dtype=np.float64)
    img -= img.mean()
    img /= img.std()
    #img = np.reshape(img, (1,256,256,3) # this is the shape our model expects
    X = np.empty((1,256,256,3))
    X[0,] = img
    predict = segmentation_model.predict(X)

    return predict.reshape(256,256)


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename 
        #img.save(img_path)

        p = predict_label(img_path)

        predicted_mask = predict_segmentation_mask(img_path)
        original_img = cv2.imread(img_path)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
        plt.axis('off')
        axes[0].imshow(original_img)
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        axes[1].imshow(predicted_mask)
        axes[1].get_xaxis().set_visible(False)
        axes[1].get_yaxis().set_visible(False)

        fig.tight_layout()
        

        seg_path = "static/seg_images/" + img.filename 
        plt.savefig(seg_path)

    return render_template("index.html", prediction = p,seg_path=seg_path)




if __name__ =='__main__':
    #app.debug = True
    app.run(debug = True)
