### Env File Structure
```
Long=
Lat=
Senec_Email=
Senec_Pass=
```

### Get all the csv files from senec dashboard. Please create a ```/data``` folder in the root.
```python
python downloadData.py
```

### Prepare the data and save to pickle file
```python
python dataloader.py
```

### Train the model
```python
python train_lstm.py # For the lstm based model
python train_mlp.py # For the mlp based model
```
 
### Use the model for inference (change the inference function call to the one you want to use)
```python
python inference.py
```

### Info
This was part of the submission for the team "solar-predict" in the bw-ki contest 2024
Username: zxp

