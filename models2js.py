import subprocess
import os

files = os.listdir('models')

for file in files:
    dirname = 'jsmodels/' + file.split('.')[0]
    try:
        os.mkdir(dirname)
    except:
        pass

    #tensorflowjs_converter --input_format keras --quantization_bytes 1 apple.h5 applejs
    subprocess.call(['tensorflowjs_converter', '--input_format', 'keras', "--quantization_bytes", '1', 'models/{}'.format(file), dirname])

