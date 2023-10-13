python3 -m pip install --no-cache-dir --upgrade pip
python3 -m pip install --no-cache-dir wheel
python3 -m pip install --no-cache-dir torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install --no-cache-dir \
    mkl \
    six \
    torchtext==0.4.0 \
    future \
    configargparse \
    cffi \
    joblib \
    librosa \
    numba==0.43.0 \
    llvmlite==0.32.1 \
    Pillow \
    pyrouge \
    opencv-python \
    git+git://github.com/NVIDIA/apex.git@700d6825e205732c1d6be511306ca4e595297070 \
    pretrainedmodels \
    filelock \
    tokenizers \
    dataclasses \
    regex \
    sentencepiece \
    sacremoses \
    torchsummaryX
