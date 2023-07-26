#*******************************************************************************
# Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
#*******************************************************************************

#!/bin/bash

#-----------------------------------------------------------------------------
#   zendnn_PT_env_setup.sh
#
#   This script does following:
#   -Checks if important env variables are declared
#   -Checks and print version informations for following:
#       -make, gcc, g++, ld, python
#   -Sets important environment variables for benchmarking:
#       -OMP_NUM_THREADS, OMP_WAIT_POLICY, OMP_PROC_BIND
#   -Calls script to gather HW, OS, Kernel, Bios information
#----------------------------------------------------------------------------

#Function to check mandatory prerequisites
function check_mandatory_prereqs() {
    if type -P make >/dev/null 2>&1;
    then
        echo "make is installed"
        echo `make -v | grep "GNU Make"`
    else
        echo "make is not installed, install make"
        return
    fi

    if type -P gcc >/dev/null 2>&1;
    then
        echo "gcc is installed"
        echo `gcc --version | grep "gcc "`
    else
        echo "gcc is not installed, install gcc"
        return
    fi

    if type -P g++ >/dev/null 2>&1;
    then
        echo "g++ is installed"
        echo `g++ --version | grep "g++ "`
    else
        echo "g++ is not installed, install g++"
        return
    fi

    if type -P ld >/dev/null 2>&1;
    then
        echo "ld is installed"
        echo `ld --version | grep "GNU ld "`
    else
        echo "ld is not installed, install ld"
        return
    fi

    if type -P python3 >/dev/null 2>&1;
    then
        echo "python3 is installed"
        echo `python3 --version`
    else
        echo "python3 is not installed, install python3"
        return
    fi
}

#Function to check optional prerequisites
function check_optional_prereqs() {
    if type -P lscpu >/dev/null 2>&1;
    then
        echo "lscpu is installed"
        echo `lscpu --version`
    else
        echo "lscpu is not installed, install lscpu"
    fi

    # Check if hwloc/lstopo-no-graphics is installed
    if type -P lstopo-no-graphics >/dev/null 2>&1;
    then
        echo "lstopo-no-graphics is installed"
        echo `lstopo-no-graphics --version`
    else
        echo "lstopo-no-graphics is not installed, install hwloc/lstopo-no-graphics"
    fi

    # Check if uname is installed
    if type -P uname >/dev/null 2>&1;
    then
        echo "uname is installed"
        echo `uname --version`
    else
        echo "uname is not installed, install uname"
    fi

    # Check if dmidecode is installed
    if type -P dmidecode >/dev/null 2>&1;
    then
        echo "dmidecode is installed"
        echo `dmidecode --version`
    else
        echo "dmidecode is not installed, install dmidecode"
    fi
}

#------------------------------------------------------------------------------
# Check if mandatory prerequisites are installed
echo "Checking mandatory prerequisites"
check_mandatory_prereqs

echo "Checking optional prerequisites"
# Check if optional prerequisites are installed
check_optional_prereqs
echo""

#------------------------------------------------------------------------------
if [ -z "$ZENDNN_LOG_OPTS" ];
then
    export ZENDNN_LOG_OPTS=ALL:0
    echo "ZENDNN_LOG_OPTS=$ZENDNN_LOG_OPTS"
else
    echo "ZENDNN_LOG_OPTS=$ZENDNN_LOG_OPTS"
fi

if [ -z "$OMP_NUM_THREADS" ];
then
    export OMP_NUM_THREADS=96
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
else
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
fi

if [ -z "$OMP_WAIT_POLICY" ];
then
    export OMP_WAIT_POLICY=ACTIVE
    echo "OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
else
    echo "OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
fi


#By default, setting ZENDNN_PT_VERSION as v1.12
export ZENDNN_PT_VERSION="1.12"
echo "ZENDNN_PT_VERSION=$ZENDNN_PT_VERSION"
#Use local copy of ZenDNN library source code when building
#pytorch wih zendnn
if [ -z "$ZENDNN_PT_USE_LOCAL_ZENDNN" ];
then
    export ZENDNN_PT_USE_LOCAL_ZENDNN=1
fi
echo "ZENDNN_PT_USE_LOCAL_ZENDNN=$ZENDNN_PT_USE_LOCAL_ZENDNN"

# Switch to enable Conv, Add fusion on users discretion. Currently it is
# safe to enable this switch for vitis_ai models (resnet50
# inception_v3) models only. By default the switch is disabled.
# enable this flag when add doesnt need broadcasting of inputs

#optimize_for_inference has to be called only after setting this flag for this fusion to happen
export ZENDNN_PT_CONV_ADD_FUSION_SAFE=0
echo "ZENDNN_PT_CONV_ADD_FUSION_SAFE=$ZENDNN_PT_CONV_ADD_FUSION_SAFE"

#If the environment variable OMP_DYNAMIC is set to true, the OpenMP implementation
#may adjust the number of threads to use for executing parallel regions in order
#to optimize the use of system resources. ZenDNN depend on a number of threads
#which should not be modified by runtime, doing so can cause incorrect execution
export OMP_DYNAMIC=FALSE
echo "OMP_DYNAMIC=$OMP_DYNAMIC"

# ZENDNN_GEMM_ALGO is set to 3
export ZENDNN_GEMM_ALGO=3
echo "ZENDNN_GEMM_ALGO=$ZENDNN_GEMM_ALGO"

# Primitive Caching Capacity
export ZENDNN_PRIMITIVE_CACHE_CAPACITY=1024
echo "ZENDNN_PRIMITIVE_CACHE_CAPACITY: $ZENDNN_PRIMITIVE_CACHE_CAPACITY"


# Enable primitive create and primitive execute logs. By default it is disabled
export ZENDNN_PRIMITIVE_LOG_ENABLE=0
echo "ZENDNN_PRIMITIVE_LOG_ENABLE: $ZENDNN_PRIMITIVE_LOG_ENABLE"

# By default build Open sourced PyTorch and ZenDNN
export ZENDNN_PYTORCH_USE_OPENSOURCE=1
#----------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Path related details
#-----------------------------------------------------------------------------
#Check if below declaration of PYTORCH_GIT_ROOT is correct
export PYTORCH_GIT_ROOT=$(pwd)
if [ -z "$PYTORCH_GIT_ROOT" ];
then
    echo "Error: Environment variable PYTORCH_GIT_ROOT needs to be set"
    echo "Error: \$PYTORCH_GIT_ROOT points to root of ZENDNN_PyTorch repo"
else
    [ ! -d "PYTORCH_GIT_ROOT" ] && echo "Directory ZenDNN_PyTorch DOES NOT exists!"
    echo "PYTORCH_GIT_ROOT: $PYTORCH_GIT_ROOT"
fi

#Change ZENDNN_PARENT_FOLDER as per need in future
#Current assumption, PyTorch is located parallel to ZenDNN
cd ..
export ZENDNN_PARENT_FOLDER=$(pwd)
cd -

if [ -z "$ZENDNN_PARENT_FOLDER" ];
then
    echo "Error: Environment variable ZENDNN_PARENT_FOLDER needs to be set"
else
    echo "ZENDNN_PARENT_FOLDER: $ZENDNN_PARENT_FOLDER"
fi

#Check if below declaration of ZENDNN_GIT_ROOT is correct
export ZENDNN_GIT_ROOT=$ZENDNN_PARENT_FOLDER/ZenDNN
if [ -z "$ZENDNN_GIT_ROOT" ];
then
    echo "Error: Environment variable ZENDNN_GIT_ROOT needs to be set"
    echo "Error: \$ZENDNN_GIT_ROOT points to root of ZENDNN repo"
else
    [ ! -d "$ZENDNN_GIT_ROOT" ] && echo "Directory ZenDNN DOES NOT exists!"
    echo "ZENDNN_GIT_ROOT: $ZENDNN_GIT_ROOT"
fi

#Change ZENDNN_UTILS_GIT_ROOT as per need in future
cd ..
export ZENDNN_UTILS_GIT_ROOT=$ZENDNN_PARENT_FOLDER/ZenDNN_utils
cd -
if [ -z "$ZENDNN_UTILS_GIT_ROOT" ];
then
    echo "Error: Environment variable ZENDNN_UTILS_GIT_ROOT needs to be set"
else
    [ ! -d "$ZENDNN_UTILS_GIT_ROOT" ] && echo "Directory ZenDNN_utils DOES NOT exists!"
    echo "ZENDNN_UTILS_GIT_ROOT: $ZENDNN_UTILS_GIT_ROOT"
fi

#Change ZENDNN_TOOLS_GIT_ROOT as per need in future
cd ..
export ZENDNN_TOOLS_GIT_ROOT=$ZENDNN_PARENT_FOLDER/ZenDNN_tools
cd -
if [ -z "$ZENDNN_TOOLS_GIT_ROOT" ];
then
    echo "Error: Environment variable ZENDNN_TOOLS_GIT_ROOT needs to be set"
else
    [ ! -d "$ZENDNN_TOOLS_GIT_ROOT" ] && echo "Directory ZenDNN_tools DOES NOT exists!"
    echo "ZENDNN_TOOLS_GIT_ROOT: $ZENDNN_TOOLS_GIT_ROOT"
fi

export PYTORCH_BENCHMARK_GIT_ROOT=$ZENDNN_PARENT_FOLDER/pytorch-benchmarks
echo "PYTORCH_BENCHMARK_GIT_ROOT: $PYTORCH_BENCHMARK_GIT_ROOT"

#-------------------------------------------------------------------------------
# HW, HW architecture, Cache, OS, Kernel details
#-----------------------------------------------------------------------------
# Go to ZENDNN_GIT_ROOT
cd $PYTORCH_GIT_ROOT

chmod u+x scripts/gather_hw_os_kernel_bios_info.sh
echo "scripts/gather_hw_os_kernel_bios_info.sh"
source scripts/gather_hw_os_kernel_bios_info.sh true > system_hw_os_kernel_bios_info.txt

#-------------------------------------------------------------------------------
echo -e "\n"
echo "Please set below environment variables explicitly as per the platform you are using!!"
echo -e "\tOMP_NUM_THREADS, GOMP_CPU_AFFINITY"
echo -e "\n"
