import os
from flask import Flask, render_template, request, jsonify

from models import UNet11, UNet16
from utils import now_str, overlay_transparent, to_heatmap


TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=TEMPLATE_DIR)

@app.route('/')
def index():
    return render_template('index.html', title='flask test', message=now_str())

@app.route('/api/upload', methods=['POST'])
def upload(key):
    print(json.loads(request.data))
    return jsonify({'data': 'hi'})

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=4000)
