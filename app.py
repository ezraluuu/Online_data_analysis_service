import pandas as pd
from flask import Flask, request
from utils import config

app = Flask(__name__)

@app.route('/regression', methods=['POST'])
def regression():  # put application's code here



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1984,
            debug=True)
