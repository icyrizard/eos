FROM smvanveen/computer-vision:20161109143812

RUN git clone https://ceres-solver.googlesource.com/ceres-solver
RUN (cd ceres-solver; make -j3)
RUN (cd ceres-solver; make install)
#
#
ENV PYTHONPATH=/usr/local/eos/bin/:$PYTHONPATH
#
WORKDIR /libs
RUN git clone https://github.com/pybind/pybind11.git
RUN (cd pybind11; mkdir build; cd build; cmake -DPYBIND11_PYTHON_VERSION=2.7 ..);
RUN (cd pybind11/build; make -j4 && make install);

# extra packages:
# graphviz: for cProfiling using pycallgraph.
# libeigen3-dev: for eos: 3D morphable face model fitting library.
RUN apt-get install -y \
    libeigen3-dev \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libeigen3-dev

WORKDIR /eos

ADD 3rdparty /eos
ADD cmake /eos
ADD python /eos

RUN mkdir /data
