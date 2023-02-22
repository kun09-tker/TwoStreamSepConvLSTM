echo "install dependencies: (use cu111 because colab has CUDA 11.1)"
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
echo "install mmcv-full thus we could use CUDA operators"
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
echo "install scipy"
pip install scipy

echo "install scipy"
pip install scipy

echo "Update numpy"
pip uninstall numpy -y
pip install numpy

echo "Install torch"
pip install torch>=1.5

echo "Install moviepy"
pip install moviepy

echo "Install imageio"
pip3 install imageio==2.4.1

echo "Install tqdm"
pip install tqdm