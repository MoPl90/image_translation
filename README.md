# Image-to-image translation for biomedical image synthesis

This Code implements the UNet segmentation network which is widely used in biomedical image segmentation [1]. This network is optimised for uses with neuroradiological images of the brain. In our work [2,3] we used this network to segment anatomical brain structes as well as stroke lesions.


## Usage
To use this code, you have to set up a python enviornment:
```
$ conda create -n yourenv python==3.7
$ conda activate yourenv
$ pip install -r requirements.txt
```
To train a UNet
```
$ python train.py -g [GPU identifier; None for CPU] \
                  -c [/path/to/settings_file.cfg] \
                  -o [output directory; inferred from settings file if ''] \
                  -t [dropout sampling 1/0; default 0] \
                  -m [multi-thread data generator 1/0; default 0] \
```
The trained model will be saved in the output directory specified as an optional argument or in the `savmodpath` defined in the settings file. There, you will also find model logs and callback outputs. 

An example setting is provided in `setting/settings_example.cfg`.


## References

[1] Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab N., Hornegger J., Wells W., Frangi A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science, vol 9351. Springer, Cham. https://doi.org/10.1007/978-3-319-24574-4_28

[2] Zopes J., Platscher M., Paganucci S., & Federau C. (2020). Multi-modal segmentation of 3D brain scans using neural networks. arXiv preprint arXiv:2008.04594.

[3] Platscher M., Zopes J., & Federau C. (2020). Image Translation for Medical Image Generation--Ischemic Stroke Lesions. arXiv preprint arXiv:2010.02745.
