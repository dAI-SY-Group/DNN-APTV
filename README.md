# Particle Regression
Sources to our paper on using deep learning for particle depth regression in fluids

## Dependencies
Particle regression depends on the following libraries:
*   Protobuf 3.0.0
*   Python-tk
*   Pillow 1.0
*   lxml
*   tf Slim (which is included in the "tensorflow/models/research/" checkout)
*   Jupyter notebook
*   Matplotlib
*   Tensorflow (>=1.12.0)
*   Cython
*   contextlib2
*   Keras==2.2.4
*   scikit-image==0.15.0
*   scikit-learn==0.20.2
*   opencv-contrib-python==3.4.0.12
*   pandas==0.23.4
*   tqdm==4.31.1

For detailed steps to install Tensorflow and Tensorflow object Detection API, follow the [Tensorflow installation instructions](https://www.tensorflow.org/install/) and [Tensorflow object Detection Insatallation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
## Particle detection
* In order to generate TFRecord file format, you need to convert your dataset to this file format.
```bash
    python3 generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
```
* The dataset (TFRecord files) and its corresponding label map. Example of how to create label maps can be found in the folder data.
  ```
  item {
  id: 1
    name: 'particle'
  }
  ```
* Configuring the Particle Detection Training Pipeline in the file `faster_rcnn_resnet101_kali.config`
 * input configuration looks as follows:
   ```
   tf_record_input_reader {
     input_path: "/data/train.record"
   }
   label_map_path: "/data/particle_classes_label_map.pbtxt"
   ```
 * The `train_config` defines parts of the training process in our case:
   ```
   train_config: {
     batch_size: 2
     #replicas_to_aggregate: 2
     batch_queue_capacity:1
     num_batch_queue_threads: 1
     prefetch_queue_capacity: 1
     optimizer {
       momentum_optimizer: {
         learning_rate: {
           manual_step_learning_rate {
           initial_learning_rate: 0.0003
           schedule {
             step: 900000
             learning_rate: .00003
           }
           schedule {
             step: 1200000
             learning_rate: .000003
             }
           }
         }
       momentum_optimizer_value: 0.9
       }
      use_moving_average: false
     }
     gradient_clipping_by_norm: 10.0
   fine_tune_checkpoint: ""
     from_detection_checkpoint: true
     num_steps: 1500000
     data_augmentation_options {
       random_horizontal_flip {
       }
     }
   }
  ```
 * In order to speed up the training process, it is recommended to reuse the pre-existing object detection checkpoint. The pre-trained checkpoints can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).


* After you created the required input file, you can train your model.
```bash
python3 train.py  --logtostderr --pipeline_config_path=/faster_rcnn_resnet101_kali.config  --train_dir=
```

* After training the model, you can get bounding box coordinates by running `particle_detection_bb.ipynb`.



# Particle regression
* Before you train the particle regression model you should crop the particle images to 180 x 180 fixed size images.
```bash
    python3 particle_crop.py
```

* You can train particle regression by running the following command:
```bash
    python3 train_regression_pos.py
```
