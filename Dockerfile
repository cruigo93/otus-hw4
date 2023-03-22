FROM cerebratech/cerebraai:1.0

RUN pip install lovely-tensors
ENV TZ=Asia/Almaty
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV PYTHONPATH $PYTHONPATH:/workspace
ENV QT_QPA_PLATFORM="offscreen"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip install connected-components-3d aim==3.14.3 pynrrd einops amqpstorm boto3 click iterative-stratification


ENV PYTHONDONTWRITEBYTECODE 1
