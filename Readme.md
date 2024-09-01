### Env File Structure
```
Long=
Lat=
Senec_Email=
Senec_Pass=
```

### Get all the csv files from senec dashboarb. Please create a ```/data``` folder in the root.
```python
python downloadData.py
```

### Prepare the data and save to pickle file
```python
python dataloader.py
```

### Train the model
```python
python train.py
```
 
### Use the model for inference
```python
python inference.py
```
