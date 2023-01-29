# Speech Emotion Recognition Web Demo

## Initialization

1. Run `python3 app.py`
2. Open "http://127.0.0.1:5000" on browser

## Add Model

1. Add your model as a new folder in `./models`
  - Make sure the folder contains keras_metadata.pb and saved_model.pb file if you use keras to train
2. Add item in `./data/models.json` including any processing information and configuration
3. Change data processing and model initialization steps in `app.py` correspondingly
