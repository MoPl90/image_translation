# Image-to-image translation for biomedical image synthesis

This Code implements a variety of image-to-image translation models. These have been used in [1] to generate synthetic biomedical images for lesion segmentation in DWI images. Moreover, we provide a GAN implementation that can be used to synthesize stroke lesions. If you use this code, please cite our work [1].

## Implemented models

- cycleGAN [2]
- pix2pix [3]
- SPADE [4]
- regular GAN 

The models can be customized with different architectures (ResNet or UNet) and parameters (2D/3D) as specified in the settings file. The model's data generator accepts dicom, nifti and numpy arrays as input.


## Usage
To use this code, you have to set up a python enviornment:
```
$ conda create -n yourenv python==3.7
$ conda activate yourenv
$ pip install -r requirements.txt
```
To train a model
```
$ python train.py -g [GPU identifier; None for CPU] \
                  -c [/path/to/settings_file.cfg] \
                  -o [output directory; inferred from settings file if ''] \
                  -t [dropout sampling 1/0; default 0] \
                  -m [multi-thread data generator 1/0; default 0] \
                  -p [path/to/pre-trained/model] \
                  -l [int identifier for pre-trained model]
```
The trained model will be saved in the output directory specified as an optional argument or in the `savmodpath` defined in the settings file. There, you will also find model logs and outputs. 

An example setting is provided in `setting/settings_example.cfg`.


## References

[1] Platscher M., Zopes J., & Federau C. (2020). Image Translation for Medical Image Generation--Ischemic Stroke Lesions. arXiv preprint arXiv:2010.02745.

[2] Zhu J. Y., Park T., Isola P., & Efros A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision (pp. 2223-2232).

[3] Isola P., Zhu J. Y., Zhou T., & Efros A. A. (2017). Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).

[4] Park T., Liu M. Y., Wang T. C., & Zhu J. Y. (2019). Semantic image synthesis with spatially-adaptive normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2337-2346).
