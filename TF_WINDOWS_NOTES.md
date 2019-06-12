# TensorFlow GPU Windows 10 Notes

## Conda Installation

- [TensorFlow GPU install instructions](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)
- Install tensorflow on conda virtual environment

## Starting Conda Virtual Environment

Launch Anaconda command line.  Anaconda PowerShell doesn't appear to work.

```bash
conda activate tensorflow_gpu
```

## VSCode Debugging

- Command palette ctrl + shift + p
- Select conda tf interpreter: Command Palette -> Python: Select Interpreter
- Command Palette -> "Terminal: Select Default Shell" select command line.  make sure tensorflow_gpu is running.
- Debug twice.  Fails first time for some reason...

## Compile Protos

- Used conda to install protoc instead of downloading.  However, both methods would probably work.

  ```bash
  conda install -c anaconda protobuf
  ```

- See [Official protobuf instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#manual-protobuf-compiler-installation-and-usage)
- In Windows get a list of all .proto files (i.e. *.proto).

  ```bash
  PS C:\tensorflow1\models\research\object_detection\protos> Get-ChildItem -include *.proto -recurse | Select -exp Name
  ```

- Copy and build command below

  ```bash
   protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\calibration.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\flexible_grid_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto
  ```

- See [TensorFlow Windows 10 tutorial for examples](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

## COCOAPI on Windows

See [philferriere cocoapi for Windows](https://github.com/philferriere/cocoapi)

```bash
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

## Set PYTHONPATH

Set in Windows system environment variable "PYTHONPATH".

```bash
C:\Users\<username>\Anaconda3;C:\Users\<username>\Anaconda3\Scripts;C:\Users\<username>\Anaconda3\Library\bin;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim;C:\tensorflow1\models\research\object_detection;
```

## Running Tutorial

- Run jupyter tutorial

```bash
jupyter notebook object_detection_tutorial.ipynb
```

- Fix matplotlib inline display:
  - Fix inline image show by putting "%matplotlib inline" at the top of the In[11] block "plt.imshow(image_np)" to work properly.  See modified object_detection_tutorial.ipynb file.
  - See [stackoverflow answer](https://stackoverflow.com/questions/19410042/how-to-make-ipython-notebook-matplotlib-plot-inline) for suggestions.
- Other Suggestions Tried (Not needed for me):
  - ~~Disable noscript for that tab.~~
  - ~~Disable AdBlock~~
  - ~~Disable Antivirus~~

## Links

- [Edureka Realtime Object Detection Youtube](https://www.youtube.com/watch?v=wh7_etX91ls&t=1230s)
- [Edureka TensorFlow Tutorial for Beginners](https://www.youtube.com/playlist?list=PL9ooVrP1hQOFJ8UZl86fYfmB1_P5yGzBT)
- [Object Detection Zoos](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- [TensorFlow Object Detection Docs](https://github.com/tensorflow/models/tree/master/research/object_detection#tensorflow-object-detection-api)

## Testing Images

- Added rename PowerShell script "renamePics.ps1" for renaming all copied images in the test_images folder to image#.jpg.
- Increment the TEST_IMAGE_PATH range to match renamed images.
  - *Note: it would also be easy to just use the files as is and update the TEST_IMAGE_PATHS to select the correct image files.*

## Jupyter

- Run Cell ```Shift + Enter```.
- Clear Output - Change to ray and back to code ```Esc R Y```.

## TODOs

- ~~Add tensor flow image processing counter and estimate time to scotthew_detection_tutorial.ipynb.~~
- ~~For my usecase only show images that have object detected.~~
  - ~~RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface~~
    - Converted to save output file.  pyplot is not needed anymore.
  - ~~Write NumPy image array to output file.~~
    - [PIL](https://pillow.readthedocs.io/en/4.2.x/reference/Image.html#PIL.Image.fromarray)
- ~~Make outputarea.less bigger. class='output output_scroll' like 100 em.~~
- ~~Get index of items over 50% match.  Then map to the Category index name and output to console~~
- Convert jupyter notebook to python script.
- Exclude Tags:
  - Car, Truck, Bench, Train, TV, potted plant, stop sign, sports ball, chair, traffic light, bus, stop sign, parking meter, giraffe, bird, fire hydrant, surfboard
- Include Tags:
  - person, bicycle, handbag, book, umbrella, backpack, skateboard, bear
- Filter by box size to remove distant objects
