FROM nvidia/cuda:9.0-devel

MAINTAINER Artur Kadurin <artur@insilico.com>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.3.31-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN apt-get install -y curl grep sed dpkg

ENV PATH /opt/conda/bin:$PATH

RUN conda install pytorch torchvision cuda90 -c pytorch

RUN conda install ipython numpy pandas scikit-learn notebook matplotlib

RUN apt-get install -y vim tmux man htop

RUN pip install progressbar2

RUN conda install -y scikit-image

RUN jupyter notebook --generate-config --allow-root && \
	echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
	echo "c.NotebookApp.password = u'sha1:3379fd89793c:4dfb6fd74c64a436b43f30dd33e9a3a68433ce52'" >> ~/.jupyter/jupyter_notebook_config.py && \
	echo "c.NotebookApp.allow_remote_access = True" >> ~/.jupyter/jupyter_notebook_config.py

EXPOSE 8765

WORKDIR /home/playground

CMD [ "/bin/bash" ]

#COPY MNIST_data /home/playground/MNIST_data
COPY mnist_conv_vae_log.ipynb /home/playground/
COPY mnist_fc_vae_log.ipynb /home/playground/
COPY utils.py /home/playground/

ENTRYPOINT jupyter notebook --no-browser --allow-root --port 8765
