# FX_forecasting


foreign exchange forecasting


Run dataset.py to create the binary dataset file for training and testing.


To train model, run command:


python3 trainer.py -e 100


trained models are saved as encoder.model and decoder.models


To run test, run command:


python3 trainer.py -t


you may modify the model directory in trainer.py


direction_correctness.py will give you an estimation of movement direction accuracy.
