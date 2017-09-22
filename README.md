# DeepSense for user identification

## Dataset

Download [HHAR dataset](https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition) (741 MB):

```sh
curl -o Activity_recognition_exp.zip 'https://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip'
```

Reference SHA256 hashsum is `d4c0c53b195b523859bf71f5a349d164c7a604a321ff6b0972fbed6e03b46582`.

Convert to TFRecord format:

```sh
python hhar_to_tfrecords.py
```
