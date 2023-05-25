# Install script for directory: /home/sbkim/project_memorymaze/memoryMaze/third_party/grpc/third_party/zlib

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/sbkim/miniconda3/envs/torchbeast")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}/home/sbkim/miniconda3/envs/torchbeast/lib/libz.so.1.2.13"
      "$ENV{DESTDIR}/home/sbkim/miniconda3/envs/torchbeast/lib/libz.so.1"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/sbkim/miniconda3/envs/torchbeast/lib/libz.so.1.2.13;/home/sbkim/miniconda3/envs/torchbeast/lib/libz.so.1")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/sbkim/miniconda3/envs/torchbeast/lib" TYPE SHARED_LIBRARY FILES
    "/home/sbkim/project_memorymaze/memoryMaze/build/lib.linux-x86_64-cpython-37/libtorchbeast/libz.so.1.2.13"
    "/home/sbkim/project_memorymaze/memoryMaze/build/lib.linux-x86_64-cpython-37/libtorchbeast/libz.so.1"
    )
  foreach(file
      "$ENV{DESTDIR}/home/sbkim/miniconda3/envs/torchbeast/lib/libz.so.1.2.13"
      "$ENV{DESTDIR}/home/sbkim/miniconda3/envs/torchbeast/lib/libz.so.1"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/sbkim/miniconda3/envs/torchbeast/lib/libz.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/sbkim/miniconda3/envs/torchbeast/lib/libz.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/sbkim/miniconda3/envs/torchbeast/lib/libz.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/sbkim/miniconda3/envs/torchbeast/lib/libz.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/sbkim/miniconda3/envs/torchbeast/lib" TYPE SHARED_LIBRARY FILES "/home/sbkim/project_memorymaze/memoryMaze/build/lib.linux-x86_64-cpython-37/libtorchbeast/libz.so")
  if(EXISTS "$ENV{DESTDIR}/home/sbkim/miniconda3/envs/torchbeast/lib/libz.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/sbkim/miniconda3/envs/torchbeast/lib/libz.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/sbkim/miniconda3/envs/torchbeast/lib/libz.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/sbkim/miniconda3/envs/torchbeast/lib/libz.a")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/sbkim/miniconda3/envs/torchbeast/lib" TYPE STATIC_LIBRARY FILES "/home/sbkim/project_memorymaze/memoryMaze/build/temp.linux-x86_64-cpython-37/grpc/third_party/zlib/libz.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/sbkim/miniconda3/envs/torchbeast/include/zconf.h;/home/sbkim/miniconda3/envs/torchbeast/include/zlib.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/sbkim/miniconda3/envs/torchbeast/include" TYPE FILE FILES
    "/home/sbkim/project_memorymaze/memoryMaze/build/temp.linux-x86_64-cpython-37/grpc/third_party/zlib/zconf.h"
    "/home/sbkim/project_memorymaze/memoryMaze/third_party/grpc/third_party/zlib/zlib.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/sbkim/miniconda3/envs/torchbeast/share/man/man3/zlib.3")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/sbkim/miniconda3/envs/torchbeast/share/man/man3" TYPE FILE FILES "/home/sbkim/project_memorymaze/memoryMaze/third_party/grpc/third_party/zlib/zlib.3")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/sbkim/miniconda3/envs/torchbeast/share/pkgconfig/zlib.pc")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/sbkim/miniconda3/envs/torchbeast/share/pkgconfig" TYPE FILE FILES "/home/sbkim/project_memorymaze/memoryMaze/build/temp.linux-x86_64-cpython-37/grpc/third_party/zlib/zlib.pc")
endif()

