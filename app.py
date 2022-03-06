from flask import Flask, render_template, request
from keras.models import load_model
from werkzeug.utils import secure_filename

import pickle

from src.util import creat_index_to_word_viceversa, extract_features1, greedySearch

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']=1

@app.route('/')

def index():
   return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def view():
    if request.method == 'POST':
        # initial_time = time.
        f = request.files['file']
        print(f.filename)
        file_name = f"static/images/{secure_filename(f.filename)}"
        f.save(file_name)
        image = extract_features1(file_name)
        image = image.reshape((1,2048))
        result = greedySearch(image)
    return render_template('upload.html', testm=result, file_name=file_name)


if __name__ == '__main__':
    app.run()
