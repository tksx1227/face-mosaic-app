import cv2
import os
import numpy as np
from flask import Flask, render_template, request


app = Flask(__name__)

# カスケード型識別機の読み込み
cascade = cv2.CascadeClassifier("./cascade_file/opencv_master_data_haarcascades_haarcascade_frontalface_default.xml")
SAVE_DIR = "./static/img/"


# モザイク処理
def mosaic(img, alpha):
    # 画像の高さと幅
    w = img.shape[1]
    h = img.shape[0]

    # 最近傍法で縮小→拡大することでモザイク加工
    img = cv2.resize(img, (int(w*alpha), int(h*alpha)))
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    return img


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        if len(img_array) != 0:
            img = cv2.imdecode(img_array, 1)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            before_path = os.path.join(SAVE_DIR + "before.png")
            after_path = os.path.join(SAVE_DIR + "after.png")

            cv2.imwrite(before_path, img)

            # 顔領域の探索
            face = cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=2, minSize=(30, 30))
            for i, (x, y, w, h) in enumerate(face):
                alpha = 1 / (8 + 0.001 * w)
                edited = mosaic(img[y:y + h, x:x + w], alpha=alpha)
                img[y:y + h, x:x + w] = edited

            cv2.imwrite(after_path, img)

            return render_template('app.html', paths=[before_path, after_path])

    return render_template('app.html')


if __name__ == "__main__":
    app.debug = True
    port = int(os.getenv("PORT", 5000))
    app.run(host="127.0.0.1", port=port)