import os
import numpy as np
import tensorflow as tf
from pydub import AudioSegment, effects
from pydub.utils import mediainfo
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import noisereduce as nr
import cv2


class DataProcessing:
  def __init__(self, labelsToInclude=[], splitDuration=8, ignoreDuration=1, transformByStft=False, hop_length=512, win_length=2048, n_mels=128, timeShape=False):
    # Hyperparameter
    self.splitDuration = splitDuration
    self.ignoreDuration = ignoreDuration
    self.transformByStft = transformByStft
    self.hop_length = hop_length
    self.win_length = win_length
    self.n_mels = n_mels
    self.timeShape = timeShape
    self.dimension = (256, 256)
    self.x_test = []
    self.sr = []
    self.recording_names = []
    

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
          # info = mediainfo(wav_path)
          
          if (audio.frame_rate != 16000):
            audio = audio.set_frame_rate(16000)
          if (audio.channels != 1):
            audio = audio.set_channels(1)
            
          sr = audio.frame_rate
          
          audio = effects.normalize(audio, headroom = 5.0) # TODO: Try other head room
          x = np.array(audio.get_array_of_samples(), dtype = 'float32')
          # x = librosa.resample(x, orig_sr=audio.frame_rate, target_sr=16000)
          
          x_list.append(x)
          sr_list.append(sr)
          recording_names.append(filename)
    
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
    self.melProcessing()
    print('   Data Process Completed!')
  
  def melProcessing(self):
    # Splitting and Padding Data
    # Split at or add padding to splitDuration (hyperparameter)
    #   if remaining duration is less than 1 sec, remove
    x_test = []
    sampling_rates = []
    recording_names = []
    
    for index, processed_x in enumerate(self.x_test):
      sr = self.sr[index]
      recording_name = self.recording_names[index]
      
      duration = len(processed_x) / sr
      size = sr * self.splitDuration

      if (duration < self.splitDuration):
        processed_x = np.pad(processed_x, (0, size - len(processed_x)), 'constant')
        
        high = self.splitDuration
        highMin = str(high // 60).rjust(2, '0')
        highSec = str(high % 60).rjust(2, '0')
        
        x_test.append(processed_x)
        sampling_rates.append(sr)
        recording_names.append((recording_name, f"00:00 - {highMin}:{highSec}"))
      elif (duration > self.splitDuration):
        count = 0
        for j in range(0, len(processed_x), size):          
          splitSection = processed_x[j:j+size]
          
          low = count * self.splitDuration
          high = low + self.splitDuration
          lowMin = str(low // 60).rjust(2, '0')
          lowSec = str(low % 60).rjust(2, '0')
          highMin = str(high // 60).rjust(2, '0')
          highSec = str(high % 60).rjust(2, '0')

          # Check if it is longer than ignoreDuration
          if (len(splitSection) > self.ignoreDuration * sr):

            # Pad audio that is shorter than splitDuration
            if (len(splitSection) < size):
              padded_x = np.pad(splitSection, (0, size - len(splitSection)), 'constant')
              
              x_test.append(padded_x)
              sampling_rates.append(sr)
              recording_names.append((recording_name, f"{lowMin}:{lowSec} - {highMin}:{highSec}"))
            else:
              x_test.append(splitSection)
              sampling_rates.append(sr)
              recording_names.append((recording_name, f"{lowMin}:{lowSec} - {highMin}:{highSec}"))
          count += 1

    # Convert to Mel-Spectrogram
    x_images = []

    for i, x in enumerate(x_test):
      # Extract Mel-Sectrogram
      if (self.transformByStft == True):
        mel_spec = librosa.feature.melspectrogram(y=x, sr=sampling_rates[i], hop_length=self.hop_length, win_length=self.win_length, n_mels=self.n_mels)
        mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
      else:
        mel_spec = librosa.feature.melspectrogram(y=x, sr=sampling_rates[i])
        mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.min)

        # Force Resize Mel-Spectrogram using image
        mel_spec = cv2.resize(mel_spec, self.dimension, interpolation=cv2.INTER_CUBIC)

      if (self.timeShape):
        mel_spec = mel_spec.T

      x_images.append(mel_spec)

    x_images = [ x for x in x_images ]
    x_images = np.asarray(x_images)
    x_images = x_images.reshape(x_images.shape[0], x_images.shape[1], x_images.shape[2], 1)
    self.x_test = x_images
    self.sr = sampling_rates
    self.recording_names = recording_names
  
  def saveMelSpectrogramImage(self, path):
    n_jobs=1
    verbose=0
    jobs = []
    png_filenames = []
    for i in range(len(self.x_test)):
      x = self.x_test[i]
      x = x.reshape(x.shape[0], x.shape[1])
      sr = self.sr[i]
      filename = self.recording_names[i][0]
      
      filename = filename[:filename.find('.')] + self.recording_names[i][1].replace(':', '').replace(' ', '') + '.png'
      png_filenames.append(filename)
      filepath = os.path.join(path, filename)
      jobs.append(joblib.delayed(self.melSpecToImageProcess)(x, sr, filepath))
    
    images = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
    
    return png_filenames
    
  
  def melSpecToImageProcess(self, x, sr, filepath):
    # print(sr, filepath)
    # librosa.display.specshow(x, sr=sr, x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    # plt.savefig(filepath)
    params = {"ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w"}
    plt.rcParams.update(params)
    
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(x, sr=sr, ax=ax, x_axis='time', y_axis='mel')
    fig.colorbar(p, format='%+2.0f dB')
    fig.savefig(filepath, transparent=True, bbox_inches='tight')
    