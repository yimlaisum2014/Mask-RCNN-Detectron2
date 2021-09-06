[Tutorial-by-Alex](https://drive.google.com/drive/folders/1Te0Z4GQ5oq1zPeyxpZb-ReG48k9kvdlQ?usp=sharing)  
  
  
Detectron2 is Facebook AI Research's next generation software system
that implements state-of-the-art object detection algorithms.
It is a ground-up rewrite of the previous version,
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>

### What's New
* It is powered by the [PyTorch](https://pytorch.org) deep learning framework.
* Includes more features such as panoptic segmentation, densepose, Cascade R-CNN, rotated bounding boxes, etc.
* Can be used as a library to support [different projects](projects/) on top of it.
  We'll open source more research projects in this way.
* It [trains much faster](https://detectron2.readthedocs.io/notes/benchmarks.html).

See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
to see more demos and learn about detectron2.

  
  
## How to run
  ```Shell
  # environment
  source environment.sh  
    
  # download model
  source download_model.sh

  # download dataset
  cd datasets
  python3 download_dataset.py   
    
  # launch  
  roslaunch rcnn_pkg mask_rcnn_prediction.launch

  # call service
  rosservice call /id_prediction_mask "obj_id: 'what object'"
  rosservice call /object "object: 'bowl'"
  rosservice call /get_pose_con "con: true"
  ```


