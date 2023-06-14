# **FontR**

# **Info about project**

The aim of this project was to create a solution for font recognition, which was an idea proposed by one team member. As a quite big amount of time passed between the initial idea and the actual implementation of the project, several scientific publications addressing similar challenges were published during that period. We familiarized ourselves with some of these publications, and based on them, we developed our own solution.

Team members:

[@madziejm](https://github.com/madziejm)
[@mleonowicz](https://github.com/mleonowicz)
[@kjpolak](https://github.com/kjpolak)

Publications on the problem of font recognition:

- Convolution Neural Networks for Arabic Font Recognition
link z abstraktem: https://ieeexplore.ieee.org/document/9067875
link do pejpera: https://sci-hub.se/10.1109/SITIS.2019.00031

- Large-Scale Visual Font Recognition (bardzo archaiczne, 2014)
link do pejpera: https://openaccess.thecvf.com/content_cvpr_2014/papers/Chen_Large-Scale_Visual_Font_2014_CVPR_paper.pdf

- Font Recognition in Natural Images via Transfer Learning
link z abstraktem: https://www.researchgate.net/publication/322424375_Font_Recognition_in_Natural_Images_via_Transfer_Learning
link do pejpera: https://sci-hub.se/10.1007/978-3-319-73603-7_19

- Farsi Font Recognition Using Holes of Letters and Horizontal Projection Profile (2011, bardziej jako ciekawostka)
link z abstraktem: https://link.springer.com/chapter/10.1007/978-3-642-27337-7_21
link do pejpera: https://sci-hub.se/10.1007/978-3-642-27337-7_21

- HENet: Forcing a Network to Think More for Font Recognition
link z abstraktem: https://dl.acm.org/doi/10.1145/3503047.3503055
link do pejpera: https://arxiv.org/pdf/2110.10872.pdf

- Convolutional Neural Networks for Font Classification
link do pejpera: https://arxiv.org/pdf/1708.03669.pdf

## **Code layout and pipeline** - highlights

```
src/
┣ fontr/
┃ ┣ fontr/
┃ ┃ ┣ __pycache__/
┃ ┃ ┣ autoencoder.py
┃ ┃ ┣ classifier.py
┃ ┃ ┣ logger.py
┃ ┃ ┣ transforms.py
┃ ┃ ┗ __init__.py
┃ ┣ pipelines/
┃ ┃ ┣ bcf_preprocessing/
┃ ┃ ┣ data_processing/
┃ ┃ ┣ data_science/
┃ ┃ ┣ __pycache__/
┃ ┃ ┣ nodes.py
┃ ┃ ┗ __init__.py
┃ ┣ __pycache__/
┃ ┃ ┣ datasets.cpython-310.pyc
┃ ┃ ┣ pipeline_registry.cpython-310.pyc
┃ ┃ ┣ settings.cpython-310.pyc
┃ ┃ ┗ __init__.cpython-310.pyc
┃ ┣ datasets.py
┃ ┣ pipeline_registry.py
┃ ┣ settings.py
┃ ┣ __init__.py
┃ ┗ __main__.py
┣ tests/
┃ ┣ pipelines/
┃ ┃ ┗ __init__.py
┃ ┣ conftest.py
┃ ┣ test_autoencoder.py
┃ ┣ test_classifier.py
┃ ┣ test_run.py
┃ ┗ __init__.py
┣ requirements-test-only.txt
┗ requirements.txt
```
![](https://hackmd.io/_uploads/rkT23Owvh.png)



## **Running**

```
poetry run pip install -r src/requirements.txt
poetry run kedro build-reqs
poetry run kedro run
```

## **Weight and Biases logging**
To use W&B logging in this project create a file named credentials_wandb.yml in the conf/local directory. It should have one attribute called api_key that stores an API Key to your W&B account.

Once you have set up the credentials_wandb.yml file, you can run the project and enjoy the benefits of W&B logging. The project will automatically log relevant experiment metrics and provide visualizations through the W&B platform.



## **AdobeVFR Dataset**

AdobeVFR dataset focuses on popular fonts and consists of synthetic and real-world data. The dataset is divided into four parts.

1. **Synthetic Data**
   - **VFR_syn_train:** This subset contains 1,000 images per class for training.
   - **VFR_syn_val:** This subset includes 100 images per class for validation.
   
   To generate synthetic training data, the dataset creators render long English words sampled from a large corpus. The resulting text images are tightly cropped, grayscale, and size-normalized. Each of the 2,383 font classes in the dataset has 1,000 training images and 100 validation images.

2. **Real-world Data**
   - **VFR_real_test:** This subset consists of 4,384 real-world test images with reliable font labels. These images were collected from typography forums where users seek help in identifying fonts. The images are converted to grayscale, manually cropped, and normalized to a height of 105 pixels. These images exhibit larger appearance variations due to scaling, background clutter, lighting, noise, perspective distortions, and compression artifacts.
   - **VFR_real_u:** This subset contains 197,396 unlabeled real-world images. These images were not annotated with font labels but were utilized to pre-train a "shared-feature" extraction subnetwork to reduce the domain gap.

Also a fontlist specifying the 2,383 font classes used in the dataset is provided, allowing to download the corresponding font files (.otf) and render synthetic images.

Furthermore, the text mentions the availability of sample codes, provided in bcf format, for processing the dataset. These sample codes are intended for use with cuda-convenet and are not optimized for product-level usage.


Examples:

![](https://hackmd.io/_uploads/ByjFx_Dwh.png)

![](https://hackmd.io/_uploads/SyungOvvh.png)

![](https://hackmd.io/_uploads/BJ9ZWuvPh.png)


## **Transfer learning**
WIP
