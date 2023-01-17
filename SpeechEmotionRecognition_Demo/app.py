import os
import shutil
import json
import cv2
import numpy as np
import tensorflow as tf
from pomegranate import *
from flask import Flask, request, render_template

from DataProcessing import DataProcessing
from HMMDataProcessing import HMMDataProcessing

UPLOAD_DIR = os.path.join('static', 'data')
MODEL_PATH = 'models'
MODEL_CONFIG_PATH = os.path.join('static', 'models.json')
MEL_SPEC_DIR = os.path.join('static', 'melSpec')

app = Flask(__name__)
app.config['UPLOAD_DIR'] = UPLOAD_DIR
app.config['MODEL_PATH'] = MODEL_PATH
app.config['MODEL_CONFIG_PATH'] = MODEL_CONFIG_PATH
app.config['MEL_SPEC_DIR'] = MEL_SPEC_DIR

modelListConfig = None

@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/models')
def models():
  with open(app.config['MODEL_CONFIG_PATH'], 'r') as f:
    global modelListConfig
    modelListConfig = json.load(f)
    modelOptions = []
    count = 0
    for modelConfig in modelListConfig:
      modelOptions.append({
        'id': count,
        'name': modelConfig['name']
      })
      count += 1
  
  return {'data': modelOptions, 'status': 'ok', 'errMsg': ''}

@app.route('/mel-spectrogram', methods=['POST'])
def melSpectrogram():
  # 1). Empty upload directory
  emptyDirectory(app.config['MEL_SPEC_DIR'])
  
  # 2). Check if model choice parameter is passed correctly
  if ('modelChoice' not in request.form or request.form['modelChoice'] == 'null'):
    errMsg = 'Model is not selected! Please select a model from dropdown!'
    print('Failed: ' + errMsg)
    return {'data': [], 'status': 'failed', 'errMsg': errMsg}
  
  # 3). Check if 
  if ('dataFileName' in request.form and request.form['dataFileName'] != 'null'):
    dataFileName = request.form['dataFileName']
    dataFileName = dataFileName[:dataFileName.find('.')] + ".wav"
  else:
    errMsg = f"File is not indicated!"
    print('Failed: ' + errMsg)
    return {'data': [], 'status': 'failed', 'errMsg': errMsg}

  # 4). B Only: Load and Process data
  modelChoice = int(request.form['modelChoice'])
  model, dataModel = getModelAndData(modelChoice, dataFileName=dataFileName)
  
  # 5). Model Prediction
  try:
    y_pred = np.argmax(model.predict(dataModel.x_test), axis=1)
  except Exception as e:
    errMsg = 'Emotion Prediction from Model Failed! ' + e
    print('Failed: ' + errMsg)
    return {'data': [], 'status': 'failed', 'errMsg': errMsg}
  
  print('Result Predicted!')
  
  # 6). Pack mel-spectrogram
  path = os.path.join('static', 'melSpec')
  try:
    png_filenames = dataModel.saveMelSpectrogramImage(path)
    png_images_data = []
    
    for i in range(len(png_filenames)):
      png_filename = png_filenames[i]
      recording_name = dataModel.recording_names[i][0]
      section = dataModel.recording_names[i][1]
      predicted_label = dataModel.labels_name[y_pred[i]]
      
      png_filepath = os.path.join(path, png_filename)
      
      if os.path.isfile(png_filepath):
        import base64
        with open(png_filepath, "rb") as img_file:
          print(png_filepath)
          image_string = base64.b64encode(img_file.read()).decode("utf-8")
          
          png_images_data.append({
            'name': recording_name,
            'section': section,
            'mel_spectrogram': image_string,
            'emotion': predicted_label
          })
      else:
        errMsg = f"Cannot find outputted mel spectrogram image in backend!'"
        print('Failed: ' + errMsg)
        return {'data': [], 'status': 'failed', 'errMsg': errMsg}
    
    print(f"Length = {len(png_images_data)}")
    return {'data': png_images_data, 'status': 'ok', 'errMsg': ''}
      
  except Exception as e:
    errMsg = 'Save mel-spectrogram image in backend failed! ' + str(e)
    print('Failed: ' + errMsg)
    return {'data': [], 'status': 'failed', 'errMsg': errMsg}
  
    
  

@app.route('/predict', methods=['POST'])
def predict():
  # 1). Empty upload directory
  emptyDirectory(app.config['UPLOAD_DIR'])
    
  # 2). Check if model choice parameter is passed correctly
  if ('modelChoice' not in request.form or request.form['modelChoice'] == 'null'):
    errMsg = 'Model is not selected! Please select a model from dropdown!'
    print('Failed: ' + errMsg)
    return {'data': [], 'status': 'failed', 'errMsg': errMsg}
  
  # 3). Get audio files and save in backend
  if (len(request.files) != 0):
    for filename in request.files:
      try:
        file = request.files[filename]
        file.save(os.path.join(app.config['UPLOAD_DIR'], file.filename))
      except Exception as e:
        errMsg = 'Save audio file in backend failed! ' + e
        print('Failed: ' + errMsg)
        return {'data': [], 'status': 'failed', 'errMsg': errMsg}
  else:
    warnMsg = 'No audio data to predict.'
    print('Warning: ' + warnMsg)
    return {'data': [], 'status': 'warning', 'errMsg': warnMsg}

  # 4). A: Get Model Choice and Configure Model; B: Load and Process data
  modelChoice = int(request.form['modelChoice'])
  print(modelChoice)
  if ((modelListConfig != None) & (modelListConfig[modelChoice]["isHMM"] == 0)):
    model, dataModel = getModelAndData(modelChoice)
  elif ((modelListConfig != None) & (modelListConfig[modelChoice]["isHMM"] == 1)):
    model, dataModel = getHMMAndData(modelChoice)  
  # 5). Model Prediction
  if (modelListConfig[modelChoice]["isHMM"] == 0):
    try:
      y_percentages = model.predict(dataModel.x_test)
      y_pred = np.argmax(y_percentages, axis=1)
    except Exception as e:
      errMsg = 'Emotion Prediction from Model Failed! ' + e
      print('Failed: ' + errMsg)
      return {'data': [], 'status': 'failed', 'errMsg': errMsg}
  elif (modelListConfig[modelChoice]["isHMM"] == 1):
    try:
      y_percentages = []
      y_pred = []
      current_index = 0
      result = model.predict_proba(dataModel.x_test)
      for _, count in enumerate(dataModel.count_list):
        y_percentage = np.array([0,0,0,0])
        for i in range(count):
            y_percentage = np.add(y_percentage, result[current_index]/count)
            current_index += 1
        y_percentages.append(y_percentage)
        y_pred.append(np.argmax(y_percentage))    
    except Exception as e:
      errMsg = 'Emotion Prediction from Model Failed! ' + e
      print('Failed: ' + errMsg)
      return {'data': [], 'status': 'failed', 'errMsg': errMsg}
  print('Result Predicted!')
  
  print(y_percentages)
  print(y_pred)
  # 5). Pack and return
  predicted_data_list = []
  for i, pred in enumerate(y_pred):
    y_percentage = y_percentages[i]
    predicted_label = dataModel.labels_name[pred]
    recording_name = dataModel.recording_names[i]
    
    percentage_dict = {}
    for pos, percent in enumerate(y_percentage):
      
      percentage_dict[dataModel.labels_name[pos]] = float(percent)
  
    predicted_data_list.append({
      'name': recording_name[0],
      'section': recording_name[1],
      'emotion': predicted_label,
      'percentage': percentage_dict
    })  
  
  return {'data': predicted_data_list, 'status': 'ok', 'errMsg': ''}

def emptyDirectory(directory):
  for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    try:
      if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
      elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
    except Exception as e:
      errMsg = f'Empty directory "{directory}" failed! ' + str(e)
      print('Failed: ' + errMsg)
      return {'data': [], 'status': 'failed', 'errMsg': errMsg}

def getModelAndData(modelChoice, dataFileName=None):
  global modelListConfig
  if (modelListConfig != None):
    if (modelChoice < len(modelListConfig)):
      modelConfig = modelListConfig[modelChoice]
      modelName = modelConfig['name']
      folderName = modelConfig['folderName']
      labelsToInclude = modelConfig['labelsToInclude']
      splitDuration = modelConfig['splitDuration']
      ignoreDuration = modelConfig['ignoreDuration']
      transformByStft = modelConfig['transformByStft']
      hop_length = modelConfig['hop_length']
      win_length = modelConfig['win_length']
      n_mels = modelConfig['n_mels']
      timeShape = modelConfig['timeShape']

      # A). Get Model
      try:
        print(f"Loading Model {modelName} from {app.config['MODEL_PATH']}/{folderName}...")
        modelDir = os.path.join(os.getcwd(), app.config['MODEL_PATH'], folderName)
        model = tf.keras.models.load_model(modelDir)
        print('   Model Loading Completed!')
      except Exception as e:
        errMsg = f"Loading model '{modelName}' Failed! " + e
        print('Failed: ' + errMsg)
        return {'data': [], 'status': 'failed', 'errMsg': errMsg}
      
      # B). Get Data Model
      try:
        dataModel = DataProcessing(labelsToInclude=labelsToInclude,
                                  splitDuration=splitDuration,
                                  ignoreDuration=ignoreDuration,
                                  transformByStft=transformByStft,
                                  hop_length=hop_length,
                                  win_length=win_length,
                                  n_mels=n_mels,
                                  timeShape=timeShape)
        dataModel.loadAndExtractTestData(app.config['UPLOAD_DIR'], dataFileName=dataFileName)
        dataModel.processData()
      except Exception as e:
        errMsg = 'Data Processing Failed! ' + str(e)
        print('Failed: ' + errMsg)
        return {'data': [], 'status': 'failed', 'errMsg': errMsg}

      return model, dataModel
    else:
      errMsg = 'Selected model not available in backed!'
      print('Failed: ' + errMsg)
      return {'data': [], 'status': 'failed', 'errMsg': errMsg}
  else:
    errMsg = 'modelListConfig variables not initialize in backend'
    print('Failed: ' + errMsg)
    return {'data': [], 'status': 'failed', 'errMsg': errMsg}

def getHMMAndData(modelChoice, dataFileName=None):
  global modelListConfig
  if (modelListConfig != None):
    if (modelChoice < len(modelListConfig)):
      modelConfig = modelListConfig[modelChoice]
      modelName = modelConfig['name']
      folderName = modelConfig['folderName']
      labelsToInclude = modelConfig['labelsToInclude']

      # A). Get Model
      try:
        print(f"Loading Model {modelName} from {app.config['MODEL_PATH']}/{folderName}...")
        modelDir = os.path.join(os.getcwd(), app.config['MODEL_PATH'], folderName, "model.json")
        with open(modelDir, 'r') as openfile:
            json_object = json.load(openfile)
        model = HiddenMarkovModel.from_json(json_object)
        print('   Model Loading Completed!')
      except Exception as e:
        errMsg = f"Loading model '{modelName}' Failed! " + e
        print('Failed: ' + errMsg)
        return {'data': [], 'status': 'failed', 'errMsg': errMsg}
      
      # B). Get Data Model
      try:
        dataModel = HMMDataProcessing(labelsToInclude=labelsToInclude)
        dataModel.loadAndExtractTestData(app.config['UPLOAD_DIR'], dataFileName=dataFileName)
        dataModel.processData()
      except Exception as e:
        errMsg = 'Data Processing Failed! ' + e
        print('Failed: ' + errMsg)
        return {'data': [], 'status': 'failed', 'errMsg': errMsg}

      return model, dataModel
    else:
      errMsg = 'Selected model not available in backed!'
      print('Failed: ' + errMsg)
      return {'data': [], 'status': 'failed', 'errMsg': errMsg}
  else:
    errMsg = 'modelListConfig variables not initialize in backend'
    print('Failed: ' + errMsg)
    return {'data': [], 'status': 'failed', 'errMsg': errMsg}

if __name__ == "__main__":
  app.run()