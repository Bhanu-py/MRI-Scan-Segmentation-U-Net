# MRI-Segmentation using U-Net
Image Segmentation of MRI scans of abdomen from cancer patients, to help cancer patients get accurate radio therapy treatment and less side effects.

The UW-Madison Carbone Cancer Center is a pioneer in MR-Linac based radiotherapy, and has treated patients with MRI guided radiotherapy based on their daily anatomy since 2015. UW-Madison has generously agreed to support this project which provides anonymized MRIs of patients treated at the UW-Madison Carbone Cancer Center. The University of Wisconsin-Madison is a public land-grant research university in Madison, Wisconsin. The Wisconsin Idea is the university's pledge to the state, the nation, and the world that their endeavors will benefit all citizens.

![image](https://user-images.githubusercontent.com/57532016/200322371-90ee1646-c7f3-4b9c-8c63-044aadc3fe2f.png)

In this figure, the tumor (pink thick line) is close to the stomach (red thick line). High doses of radiation are directed to the tumor while avoiding the stomach. The dose levels are represented by the rainbow of outlines, with higher doses represented by red and lower doses represented by green.

## Dataset Description
In this competition we are segmenting organs cells in images. The training annotations are provided as RLE-encoded masks, and the images are in 16-bit grayscale PNG format.

Each case in this competition is represented by multiple sets of scan slices (each set is identified by the day the scan took place). Some cases are split by time (early days are in train, later days are in test) while some cases are split by case - the entirety of the case is in train or test. The goal of this competition is to be able to generalize to both partially and wholly unseen cases.

Note that, in this case, the test set is entirely unseen. It is roughly 50 cases, with a varying number of days and slices, as seen in the training set.

### Columns
- id - unique identifier for object
- class - the predicted class for the object
- EncodedPixels - RLE-encoded pixels for the identified object

## Model Architecture
![image](https://user-images.githubusercontent.com/57532016/200323664-d331ad55-ba3f-49b6-b2ba-ca68f6cc5092.png)

## References
Zhou, F., Ye, Y. & Song, Y. Image Segmentation of Rectal Tumor Based on Improved U-Net Model with Deep Learning. J Sign Process Syst (2021).
doi - https://doi.org/10.1007/s11265-021-01710-x
