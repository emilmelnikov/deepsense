## Base environment configuration ##

# Install CUDA.
curl -L -o cuda-repo.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo.deb
rm cuda-repo.deb
sudo apt-get update
sudo apt-get install -y cuda

# CUDA environment variables.
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
export CUDA_HOME=/usr/local/cuda
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
export PATH=$CUDA_HOME/bin:$PATH

# Install Anaconda.
curl -L -o anaconda.sh https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
bash anaconda.sh -b
rm anaconda.sh

# Anaconda environment variables.
export ANACONDA_HOME=$HOME/anaconda3
echo 'export ANACONDA_HOME=$HOME/anaconda3' >> ~/.bashrc
export PATH=$ANACONDA_HOME/bin:$PATH
echo 'export PATH=$ANACONDA_HOME/bin:$PATH' >> ~/.bashrc

# Install cuDNN (custom Dropbox link, since downloading cuDNN requires registration).
curl -L -o libcudnn6.deb 'https://www.dropbox.com/s/i951jplzjox2uv7/libcudnn6_6.0.21-1%2Bcuda8.0_amd64.deb?dl=0'
curl -L -o libcudnn6-dev.deb 'https://www.dropbox.com/s/6entpqhu9asz1z1/libcudnn6-dev_6.0.21-1%2Bcuda8.0_amd64.deb?dl=0'
sudo dpkg -i libcudnn6.deb libcudnn6-dev.deb
rm libcudnn6.deb libcudnn6-dev.deb

# Install additional libraries for TensorFlow.
sudo apt-get install -y libcupti-dev



## TensorFlow and Python3 ##
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp36-cp36m-linux_x86_64.whl



## TensorFlow and Python3 from sources ##

# Install TensorFlow build dependencies.
sudo apt-get install -y openjdk-8-jdk
echo 'deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8' | sudo tee /etc/apt/sources.list.d/bazel.list
curl -L https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install --assume-yes bazel

# Install TensorFlow from sources.
git clone --branch r1.3 --depth 1 https://github.com/tensorflow/tensorflow
pushd tensorflow
./configure
# Default options, except:
# * Use MKL: Y
# * Use CUDA: Y
# * cuDNN location: /usr/lib/x86_64-linux-gnu
tmux new-session -s tfbuild -A 'bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package'
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
popd
sudo --set-home env "PATH=$PATH" sh -c 'pip install /tmp/tensorflow_pkg/tensorflow-1.3.0-cp36-cp36m-linux_x86_64.whl'
conda update libgcc  # The current Anadonda version contains old version of libstdc++.



## TensorFlow and Python2 ##
conda create -n tf-py2 python=2.7
source activate tf-py2
conda install matplotlib pillow  # pillow is required for JPEG graphs.
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp27-none-linux_x86_64.whl



## DeepSense ##
git clone https://github.com/yscacaca/DeepSense
cd DeepSense
wget -O sepHARData_a.tar.gz 'https://www.dropbox.com/s/z7zpnwh2ndthd2n/sepHARData_a.tar.gz?dl=0'
tar -xzf sepHARData_a.tar.gz
find sepHARData_a -name '._*' | xargs rm  # Remove junk archive files.
ls -1 sepHARData_a/train | head | xargs -I % mv sepHARData_a/train/% eval  # Use some of the training data as test data.
tmux new-session -s deepsense -A 'python deepSense_HHAR_tf.py; sh -i'
