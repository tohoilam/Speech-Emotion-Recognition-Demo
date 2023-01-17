import os
import numpy as np
import tensorflow as tf
from pydub import AudioSegment, effects
import librosa
import joblib
import noisereduce as nr
import cv2
import speechpy
from pomegranate import *

class HMMDataProcessing:
  def __init__(self, labelsToInclude=[]):
    # Hyperparameter
    self.x_test = []
    self.sr = []
    self.recording_names = []
    self.count_list = []

    if (labelsToInclude == []):
      self.labels_name = ['Neutral', 'Frustration', 'Anger', 'Sadness', 'Happiness', 'Excitement', 'Surprise', 'Disgust', 'Fear']
    else:
      self.labels_name = labelsToInclude
      
  def loadAndExtractTestData(self, path, dataFileName=None):
    x_list = []
    sr_list = []
    recording_names = []
    
    data_path = os.path.join(os.getcwd(), path)
    
    # Convert audio file to wav format
    for dirname, _, filenames in os.walk(data_path):
      for filename in filenames:
        
        if (filename == 'desktop.ini' or filename == 'desktop.in.txt' or filename == '.DS_Store' or filename == '.DS'):
          continue
        
        if (filename[-4:] != '.wav'):
          original_file = os.path.join(dirname, filename)
          wav_filename = os.path.join(dirname, filename[:-4] + ".wav")
          if (filename[-4:] == '.m4a'):
            track = AudioSegment.from_file(original_file,  format='m4a')
            file_handle = track.export(wav_filename, format='wav')
          elif (filename[-4:] == '.mp3'):
            track = AudioSegment.from_mp3(original_file)
            file_handle = track.export(wav_filename, format='wav')
          elif (filename[-4:] == '.ogg' or filename[-5:] == '.opus'):
            track = AudioSegment.from_ogg(original_file)
            file_handle = track.export(wav_filename, format='wav')
          elif (filename[-3:] == '.au'):
            track = AudioSegment.from_file(original_file,  format='au')
            file_handle = track.export(wav_filename, format='wav')
    
    # Delete non wav file
    # Convert audio file to wav format
    for dirname, _, filenames in os.walk(data_path):
      for filename in filenames:
        
        if (filename == 'desktop.ini' or filename == 'desktop.in.txt' or filename == '.DS_Store' or filename == '.DS'):
          continue
        
        if (filename[-4:] != '.wav'):
          file_path = os.path.join(dirname, filename)
          os.remove(file_path)
    
    if (dataFileName != None):
      dataFileName = dataFileName[:dataFileName.find('.')] + '.wav'
    
    # Load and extract audio
    for dirname, _, filenames in os.walk(data_path):
      for filename in filenames:
        
        if (filename == 'desktop.ini' or filename == 'desktop.in.txt' or filename == '.DS_Store' or filename == '.DS'):
          continue
        
        if (dataFileName == None or filename == dataFileName):
          # Load Audio and x
          wav_path = os.path.join(dirname, filename)
          audio = AudioSegment.from_file(wav_path)
          if (audio.frame_rate != 16000):
            audio = audio.set_frame_rate(16000)
          sr = audio.frame_rate
          x = np.array(audio.get_array_of_samples(), dtype = 'float32')
          
          x_list.append(x)
          sr_list.append(sr)
          recording_names.append((filename, "All"))
    
    self.extractTestData(x_list, sr_list, recording_names)
  
  def extractTestData(self, x_list, sr_list, recording_names):
    print('Loading and Extracting Data...')
    
    # Process Audio
    for i, x in enumerate(x_list):
      sr = sr_list[i]
      recording_name = recording_names[i]

      processed_x, _ = librosa.effects.trim(x, top_db = 30)
      processed_x = nr.reduce_noise(processed_x, sr=sr)
      self.x_test.append(processed_x)
      self.sr.append(sr)
      self.recording_names.append(recording_name)
    
    print('   Data Loading and Extraction Completed!')
  
  def processData(self):
    print('Processing data...')
    self.featureExtraction()
    print('   Data Process Completed!')
    
  def featureExtraction(self):
    x_test = []
    sampling_rates = []
    count_list = []
    
    for index, processed_x in enumerate(self.x_test):
      sr = self.sr[index]
      mfcc = librosa.feature.mfcc(processed_x, sr=sr, n_mfcc=15, n_fft=int(0.025*sr), hop_length=int(0.01*sr))
      mfcc_normalized = speechpy.processing.cmvn(mfcc.T)
      delta = librosa.feature.delta(mfcc_normalized.T)
      deltadelta = librosa.feature.delta(delta)
      f0, _ , voice_prob = librosa.pyin(processed_x,librosa.note_to_hz('C2'),librosa.note_to_hz('C7'), sr, win_length=int(0.025*sr), hop_length=int(0.01*sr), fill_na=0)
      reduce_index = np.where(voice_prob > 0.25)[0]
      feature = np.concatenate((mfcc_normalized, delta.T, deltadelta.T, f0.reshape(f0.shape[0], 1), voice_prob.reshape(voice_prob.shape[0], 1)), axis=1)
      feature = np.take(feature, reduce_index, axis=0)
      if (len(x_test) == 0):
        x_test = feature
      else:
        x_test = np.concatenate((x_test, feature), axis=0)
      count_list.append(feature.shape[0])
    
    self.x_test = x_test
    self.count_list = count_list