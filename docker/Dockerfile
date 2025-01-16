# Determined AI's base image
FROM harbor.lins.lab/determinedai/environments:cuda-11.8-pytorch-2.0-gpu-mpi-0.31.1
# Another one of their base images, with newer CUDA and pytorch
# FROM determinedai/environments:cuda-11.8-pytorch-2.0-gpu-mpi-0.27.1
# You can check out their images here: https://hub.docker.com/r/determinedai/environments/

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1

# Update package sources and install dependencies
RUN sed -i "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list && \
    sed -i "s/security.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        unzip \
        python3-opencv \
        graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy configuration files
COPY environment.yml /tmp/environment.yml
COPY requirements.txt /tmp/requirements.txt

# Configure conda and install dependencies
RUN conda install -y conda=23.11.0 && \
    conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/conda-forge/ && \
    conda config --set show_channel_urls yes && \
    # Install packages directly instead of using env update
    conda install -y python=3.8 \
        pytorch>=2.0.0 \
        torchvision>=0.15.0 \
        cudatoolkit=11.8 \
        numpy>=1.21.0 \
        scipy>=1.9.0 \
        scikit-learn>=1.0.0 \
        matplotlib>=3.4.0 \
        seaborn>=0.12.0 \
        pillow>=8.0.0 \
        tqdm>=4.65.0 \
        pandas>=1.5.0 \
        h5py>=3.8.0 && \
    conda clean --all --force-pkgs-dirs --yes && \
    pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple && \
    pip install --no-cache-dir -r /tmp/requirements.txt