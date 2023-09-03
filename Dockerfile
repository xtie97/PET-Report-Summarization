FROM nvcr.io/nvidia/pytorch:21.03-py3
WORKDIR /UserData
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install numpy==1.23.4 matplotlib==3.4.2 torch==1.13.1 torchvision torchtext transformers==4.15.0 huggingface-hub fastai==2.2.5 bert-score ohmeow-blurr pytorch-lightning pandas==1.5.3 tqdm==4.57.0 opencv-python==4.5.5.64 scikit-learn==1.0.2 scikit-image==0.18.1 Pillow


