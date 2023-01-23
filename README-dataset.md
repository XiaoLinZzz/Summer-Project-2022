# About Dataset

Because we have reconstructed the dataset in our own way, so there are some details that need to be noted. 
More details can be found in the [build_dataset.py]().

## Description of the data

Make sure your dataset architecture is as follows :

```
Datasets/
  -Sub Dir/
    -DataFile1
    -DataFile2
    -...
  -Sub Dir/
    -DataFile1
    -DataFile2
    -...
  -...
```

### Example

<!-- In this research, we use the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset as an example. -->

In this example, the dataset is stored in the **./animals/** directory. The **./animals/** directory contains 2 subdirectories, each of which contains 5 images. The name of each subdirectory is the name of the category. The name of each image is the name of the category plus the serial number of the image. The format of each image is **.jpg**.

Because transformers require lots data to train, so we suggest you to use large dataset.

```
animals/
  -cats/
    -cats_00001.jpg
    -cats_00002.jpg
    -...
    -cats_00005.jpg
  -dogs/
    -dogs_00001.jpg
    -dogs_00002.jpg
    -...
    -dogs_00005.jpg
```

### And file formats

Please notice that the files format should be **.jpg** files. If you have other format files, please convert them to **.jpg** files.

```
- 10 images, format jpg.
```



