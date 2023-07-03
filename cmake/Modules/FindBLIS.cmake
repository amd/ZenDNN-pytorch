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

# - Find BLIS library
#
# This module sets the following variables:
#  BLIS_FOUND - set to true if a library implementing CBLAS interface is found.
#  BLIS_INCLUDE_DIR - path to include dir.
#  BLIS_LIBRARIES - list of libraries for BLIS.
#
# CPU only Dockerfile to build with AMD BLIS is available at the location
# pytorch/docker/pytorch/cpu-blis/Dockerfile
##

IF (NOT BLIS_FOUND)

add_custom_target(libamdblis ALL
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/libblis-mt.a
)

add_custom_command(
    OUTPUT
       ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/libblis-mt.a
   WORKING_DIRECTORY
       ${CMAKE_CURRENT_SOURCE_DIR}/third_party/blis
   COMMAND
       make clean && make distclean && CC=gcc  ./configure --prefix=${CMAKE_CURRENT_SOURCE_DIR}/build/blis_gcc_build  --enable-threading=openmp --enable-cblas amdzen && make -j install
   COMMAND
       cd ${CMAKE_CURRENT_SOURCE_DIR}/build
   COMMAND
       cp blis_gcc_build/lib/libblis-mt.a ${CMAKE_CURRENT_SOURCE_DIR}/build/lib
   COMMAND
       cp -r blis_gcc_build/include/blis/* blis_gcc_build/include
)

SET(BLIS_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/build/blis_gcc_build/include
)

SET(BLIS_LIB_SEARCH_PATHS
  ${CMAKE_CURRENT_SOURCE_DIR}/build/lib
)

LIST(APPEND BLIS_LIBRARIES ${BLIS_LIB_SEARCH_PATHS}/libblis-mt.a)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(BLIS DEFAULT_MSG BLIS_INCLUDE_DIR BLIS_LIBRARIES)

MARK_AS_ADVANCED(
        BLIS_INCLUDE_DIR
        BLIS_LIBRARIES
        blis-mt
)

find_package(Git)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} -c log.showSignature=false log --no-abbrev-commit --oneline -1 --format=%H
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/blis
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE BLIS_VERSION_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT GIT_FOUND OR RESULT)
    set(BLIS_VERSION_HASH "N/A")
endif()

SET(BLIS_FOUND ON)
IF(BLIS_FOUND)
        IF(NOT BLIS_FIND_QUIETLY)
                MESSAGE(STATUS "Found BLIS libraries: ${BLIS_LIBRARIES}")
                MESSAGE(STATUS "Found BLIS include: ${BLIS_INCLUDE_DIR}")
        ENDIF()
ELSE(BLIS_FOUND)
        MESSAGE(FATAL_ERROR "Could not find BLIS")
ENDIF(BLIS_FOUND)
ENDIF (NOT BLIS_FOUND)

