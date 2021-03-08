#FROM continuumio/anaconda3

FROM gautam81/anaconda3


#FROM yorek/anaconda3-windowservercore

#RUN mkdir /docker_image_bananacollector

# set a directory for the app
#WORKDIR E:/deep-reinforcement-learning/python/docker_image_bananacollector

WORKDIR C:/docker_image_bananacollector

#ADD ./sim/. /sim/


#RUN conda install -c anaconda powershell_shortcut
# RUN conda install -c intel tbb
#RUN conda install -c anaconda libsodium

#RUN conda install -c anaconda lxml


############################################################################
############################################################################
##############################################################################

# RUN conda install -c anaconda winpty

# RUN conda install -c anaconda vs2015_runtime

# RUN conda install -c anaconda menuinst

# RUN conda install -c msys2 msys2-conda-epoch

# RUN conda install -c anaconda pywin32

# RUN conda install -c anaconda comtypes

# #RUN conda install -c conda-forge comtypes

# #RUN conda install -c auto comtypes

# RUN conda install -c intel icc_rt

# #RUN conda install -c anaconda icc_rt

# RUN conda install -c conda-forge m2w64-gcc-libs-core

# RUN conda install -c conda-forge pywinpty

# RUN conda install -c anaconda vc

# #RUN conda install -c conda-forge vc

# RUN conda install -c msys2 m2w64-gmp

# RUN conda install -c anaconda win_unicode_console

# #RUN conda install -c conda-forge win-unicode-console


# RUN conda install -c anaconda xlwings


# RUN conda install -c msys2 m2w64-gcc-libgfortran


# RUN conda install -c anaconda pyreadline


# RUN conda install -c conda-forge wincertstore

# RUN conda install -c conda-forge win_inet_pton


# RUN conda install -c msys2 m2w64-libwinpthread-git

# RUN conda install -c conda-forge pywin32-ctypes

# #RUN conda install -c esri pywin32-ctypes

# RUN conda install -c anaconda powershell_shortcut



# #RUN pip install -i https://pypi.anaconda.org/ales-erjavec/simple pyreadline



############################################################################
############################################################################
##############################################################################


#RUN pip install pywinpty


# Create the environment:

#COPY Bonsai_Platform.yml .


COPY Bonsai_Platform.yml .


#RUN import gc

#RUN gc.collect()

RUN conda env create -f Bonsai_Platform.yml


# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "bonsai_env", "/bin/bash", "-c"]


# Make sure the environment is activated:
#RUN echo "Make sure libraries are installed:"
# RUN python -c "import datetime"
# RUN python -c "import json"
# RUN python -c "import os"
# RUN python -c "import dotenv"





# The code to run when container is started:
COPY Bonsai_Add_CustomSim.py .
COPY NN_Model.py .
COPY DeepQN_Agent.py .
COPY unityagents.py .
COPY brain.py .
COPY communicator.py .
COPY curriculum.py .
COPY environment.py .
COPY exception.py .
COPY rpc_communicator.py .
COPY socket_communicator.py .
COPY communicator_objects.py .
COPY agent_action_proto_pb2.py .
COPY agent_info_proto_pb2.py .
COPY brain.py .
COPY brain_parameters_proto_pb2.py .
COPY brain_type_proto_pb2.py .
COPY command_proto_pb2.py .
COPY communicator.py .
COPY curriculum.py .
COPY engine_configuration_proto_pb2.py .
COPY environment.py .
COPY environment_parameters_proto_pb2.py .
COPY exception.py .
COPY header_pb2.py .
COPY resolution_proto_pb2.py .
COPY space_type_proto_pb2.py .
COPY unity_input_pb2.py .
COPY unity_message_pb2.py .
COPY unity_output_pb2.py .
COPY unity_rl_initialization_input_pb2.py .
COPY unity_rl_initialization_output_pb2.py .
COPY unity_rl_input_pb2.py .
COPY unity_rl_output_pb2.py .
COPY unity_to_external_pb2.py .
COPY unity_to_external_pb2_grpc.py .



# RUN mkdir /sim_docker

# COPY sim/*  /sim_docker/

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "bonsai_env", "python", "Bonsai_Add_CustomSim.py"]