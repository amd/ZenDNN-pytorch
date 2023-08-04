#******************************************************************************
# Modifications Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#******************************************************************************

IF (NOT ZENDNN_FOUND)

file(GLOB zendnn_src_common_cpp "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/common/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/gemm/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/gemm/f32/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/gemm/s8x8s32/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/matmul/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/reorder/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/rnn/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/brgemm/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/amx/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/bf16/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/f32/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/gemm/s8x8s32/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/injectors/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/lrn/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/matmul/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/prelu/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/rnn/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/src/cpu/x64/shuffle/*.cpp")

set(GENERATED_CXX_ZEN
    ${zendnn_src_common_cpp}
  )

IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
   SET(BUILD_FLAG 0)
ELSE()
   SET(BUILD_FLAG 1)
ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")

add_custom_target(libamdZenDNN ALL
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/libamdZenDNN.a
)

add_custom_command(
   OUTPUT
        ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/libamdZenDNN.a
   WORKING_DIRECTORY
       ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN
   COMMAND
      make -j ZENDNN_BLIS_PATH=${CMAKE_CURRENT_SOURCE_DIR}/build/blis_gcc_build AOCC=0 LPGEMM=1 ARCHIVE=1 RELEASE=${BUILD_FLAG}
   COMMAND
       cp _out/lib/libamdZenDNN.a ${CMAKE_CURRENT_SOURCE_DIR}/build/lib
   DEPENDS
        ${zendnn_src_common_cpp}
   COMMAND
        make clean
)

add_dependencies(libamdZenDNN libamdblis)

SET(ZENDNN_INCLUDE_SEARCH_PATHS
 /usr/include
 /usr/local/include/
 /usr/local/include/zendnn/include
 /usr/local/opt/include
 /opt/include
 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN
 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/inc
 ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN/include
)
FIND_PATH(ZENDNN_INCLUDE_DIR NAMES zendnn_config.h zendnn.h zendnn_types.h zendnn_debug.h zendnn_version.h PATHS ${ZENDNN_INCLUDE_SEARCH_PATHS})
IF(NOT ZENDNN_INCLUDE_DIR)
	MESSAGE(STATUS "Could not find ZENDNN include.")
	RETURN()
ENDIF(NOT ZENDNN_INCLUDE_DIR)

SET(ADEEP_ROOT "${PROJECT_SOURCE_DIR}/third_party/adeep")
FIND_PATH(ADEEP_INCLUDE_DIR adeep.hpp PATHS ${ADEEP_ROOT} PATH_SUFFIXES include)
IF (NOT ADEEP_INCLUDE_DIR)
MESSAGE(FATAL_ERROR "ADEEP include files not found!")
RETURN()
ENDIF(NOT ADEEP_INCLUDE_DIR)
LIST(APPEND ZENDNN_INCLUDE_DIR ${ADEEP_INCLUDE_DIR})

SET(ZENDNN_LIB_SEARCH_PATHS
 ${CMAKE_CURRENT_SOURCE_DIR}/build/lib
)

LIST(APPEND ZENDNN_LIBRARIES ${ZENDNN_LIB_SEARCH_PATHS}/libamdZenDNN.a)

MARK_AS_ADVANCED(
	ZENDNN_INCLUDE_DIR
	ZENDNN_LIBRARIES
        amdZenDNN
)

find_package(Git)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} -c log.showSignature=false log --no-abbrev-commit --oneline -1 --format=%H
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNN
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE ZENDNN_PT_VERSION_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT GIT_FOUND OR RESULT)
    set(ZENDNN_PT_VERSION_HASH "N/A")
endif()

SET(ADEEP_ROOT "${PROJECT_SOURCE_DIR}/third_party/adeep")
FIND_PATH(ADEEP_INCLUDE_DIR adeep.hpp PATHS ${ADEEP_ROOT} PATH_SUFFIXES include)
IF (NOT ADEEP_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "ADEEP include files not found!")
  RETURN()
ENDIF(NOT ADEEP_INCLUDE_DIR)
LIST(APPEND ZENDNN_INCLUDE_DIR ${ADEEP_INCLUDE_DIR})

#
IF(NOT USE_OPENMP)
    MESSAGE(FATAL_ERROR "ZenDNN requires OMP library")
    RETURN()
ENDIF()

IF(USE_TBB)
  MESSAGE(FATAL_ERROR "ZenDNN requires blis library, set USE_TBB=0")
  RETURN()
ENDIF(USE_TBB)

IF(NOT BLIS_FOUND)
    FIND_PACKAGE(BLIS)
ENDIF(NOT BLIS_FOUND)

IF(BLIS_FOUND)
  LIST(APPEND ZENDNN_LIBRARIES ${BLIS_LIBRARIES})
ELSE(BLIS_FOUND)
	MESSAGE(FATAL_ERROR "ZenDNN requires blis library.")
	RETURN()
ENDIF(BLIS_FOUND)

SET(ZENDNN_FOUND ON)
IF (ZENDNN_FOUND)
	IF (NOT ZENDNN_FIND_QUIETLY)
		MESSAGE(STATUS "Found ZENDNN libraries: ${ZENDNN_LIBRARIES}")
		MESSAGE(STATUS "Found ZENDNN include: ${ZENDNN_INCLUDE_DIR}")
	ENDIF (NOT ZENDNN_FIND_QUIETLY)
ELSE (ZENDNN_FOUND)
	IF (ZENDNN_FIND_REQUIRED)
		MESSAGE(FATAL_ERROR "Could not find ZENDNN")
	ENDIF (ZENDNN_FIND_REQUIRED)
ENDIF (ZENDNN_FOUND)

ENDIF (NOT ZENDNN_FOUND)
