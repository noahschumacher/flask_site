## Server Flask File

from flask import Flask, render_template, request, jsonify, Response
import pickle

## Create the app object that will route our calls
app = Flask(__name__)


@app.route('/', methods = ['GET'])

def home():
	return '<p> This is Noah dope ass website </p>'

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=3333, debug=True)


