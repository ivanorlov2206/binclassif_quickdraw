# Default Flask app :)

import os
import random
import urllib.request

import run
from flask import Flask, render_template, request, send_file

app = Flask(__name__)

files = [x.split('.')[0] for x in os.listdir('data')]

def gen_random_name_png(ln):
    return ''.join([str(random.randint(0, 9)) for x in range(ln)]) + ".png"


@app.route('/')
def index():
    return render_template("index.html", names=files)

@app.route("/class", methods=["POST"])
def classif():
    arch = request.args.get("model", "apple")
    print(arch)
    data = request.data.decode()
    resp = urllib.request.urlopen(data)
    fname = gen_random_name_png(15)
    with open(fname, "wb") as f:
        f.write(resp.file.read())
        f.close()
    qd = run.QuickDraw(arch)
    res = qd.classif(fname)
    qd.clear()
    os.remove(fname)
    return res

app.run(debug=True)
