# particleRegression
Sources to our paper on using deep learning for particle depth regression in fluids

# particle detection
* In order to generate TFRecord file format, you need to convert your dataset to this file format.
```python
    python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
```
* The dataset (TFRecord files) and its corresponding label map. Example of how to create label maps can be found here.
```python
item {
id: 1
name: 'particle'
}
```
* After you created the required input file, you can train your model.
```python
python3 object_detection/legacy/train.py --logtostderr --pipeline_config_path=/faster_rcnn_resnet101_kali_png.config  --train_dir=
```
* After training the model, you can get bounding box coordinates by running particle_detection.ipynb.



# Particle regression
* You can train particle regression by running the following command:
```python
    python train_regression_pos.py
