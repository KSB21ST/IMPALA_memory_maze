# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


set(CPACK_BINARY_DEB "OFF")
set(CPACK_BINARY_FREEBSD "OFF")
set(CPACK_BINARY_IFW "OFF")
set(CPACK_BINARY_NSIS "OFF")
set(CPACK_BINARY_RPM "OFF")
set(CPACK_BINARY_STGZ "ON")
set(CPACK_BINARY_TBZ2 "OFF")
set(CPACK_BINARY_TGZ "ON")
set(CPACK_BINARY_TXZ "OFF")
set(CPACK_BINARY_TZ "ON")
set(CPACK_BUILD_SOURCE_DIRS "/home/sbkim/project_memorymaze/memoryMaze;/home/sbkim/project_memorymaze/memoryMaze/build/temp.linux-x86_64-cpython-37")
set(CPACK_CMAKE_GENERATOR "Ninja")
set(CPACK_COMPONENTS_ALL "Devel;Library;Tools")
set(CPACK_COMPONENT_Devel_DESCRIPTION "Development files for compiling against c-ares.")
set(CPACK_COMPONENT_Devel_DISPLAY_NAME "c-ares Development Files")
set(CPACK_COMPONENT_Devel_REQUIRED "0")
set(CPACK_COMPONENT_Library_DESCRIPTION "The c-ares binary library.")
set(CPACK_COMPONENT_Library_DISPLAY_NAME "c-ares Library")
set(CPACK_COMPONENT_Library_REQUIRED "1")
set(CPACK_COMPONENT_Tools_DESCRIPTION "Tools for using c-ares.")
set(CPACK_COMPONENT_Tools_DISPLAY_NAME "c-ares Tools")
set(CPACK_COMPONENT_Tools_REQUIRED "0")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_DEBIAN_COMPRESSION_TYPE "xz")
set(CPACK_DEBIAN_DEVEL_PACKAGE_DEPENDS "cmake (>= 3.1.0), libcares (>= 1.17.2)")
set(CPACK_DEBIAN_DEVEL_PACKAGE_NAME "c-ares-dev")
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_DEBIAN_LIBRARY_PACKAGE_NAME "libcares")
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://c-ares.haxx.se/")
set(CPACK_DEBIAN_PACKAGE_RELEASE "1")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS "1")
set(CPACK_DEBIAN_TOOLS_PACKAGE_DEPENDS "libcares (>= 1.17.2)")
set(CPACK_DEBIAN_TOOLS_PACKAGE_NAME "c-ares-tools")
set(CPACK_DEBIAN_TOOLS_PACKAGE_SHLIBDEPS "OFF")
set(CPACK_DEB_COMPONENT_INSTALL "1")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_FILE "/home/sbkim/miniconda3/envs/torchbeast/lib/python3.7/site-packages/cmake/data/share/cmake-3.26/Templates/CPack.GenericDescription.txt")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY "torchbeast built using CMake")
set(CPACK_DMG_SLA_USE_RESOURCE_FILE_LICENSE "ON")
set(CPACK_GENERATOR "STGZ;TGZ;TZ")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY "1")
set(CPACK_INSTALL_CMAKE_PROJECTS "/home/sbkim/project_memorymaze/memoryMaze/build/temp.linux-x86_64-cpython-37;torchbeast;ALL;/")
set(CPACK_INSTALL_PREFIX "/home/sbkim/miniconda3/envs/torchbeast")
set(CPACK_MODULE_PATH "/home/sbkim/project_memorymaze/memoryMaze/third_party/grpc/cmake/modules")
set(CPACK_NSIS_DISPLAY_NAME "c-ares")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
set(CPACK_NSIS_PACKAGE_NAME "c-ares")
set(CPACK_NSIS_UNINSTALL_NAME "Uninstall")
set(CPACK_OBJCOPY_EXECUTABLE "/usr/bin/objcopy")
set(CPACK_OBJDUMP_EXECUTABLE "/usr/bin/objdump")
set(CPACK_OUTPUT_CONFIG_FILE "/home/sbkim/project_memorymaze/memoryMaze/build/temp.linux-x86_64-cpython-37/CPackConfig.cmake")
set(CPACK_PACKAGE_ARCHITECTURE "x86_64")
set(CPACK_PACKAGE_CONTACT "https://c-ares.haxx.se/")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION_FILE "/home/sbkim/miniconda3/envs/torchbeast/lib/python3.7/site-packages/cmake/data/share/cmake-3.26/Templates/CPack.GenericDescription.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A C library for asynchronous DNS requests")
set(CPACK_PACKAGE_FILE_NAME "c-ares-1.17.2-Linux")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://c-ares.haxx.se/")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "c-ares")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "c-ares")
set(CPACK_PACKAGE_NAME "c-ares")
set(CPACK_PACKAGE_RELEASE "1")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR "Daniel Stenberg")
set(CPACK_PACKAGE_VERSION "1.17.2")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "17")
set(CPACK_PACKAGE_VERSION_PATCH "2")
set(CPACK_READELF_EXECUTABLE "/usr/bin/readelf")
set(CPACK_RESOURCE_FILE_LICENSE "/home/sbkim/project_memorymaze/memoryMaze/third_party/grpc/third_party/cares/cares/LICENSE.md")
set(CPACK_RESOURCE_FILE_README "/home/sbkim/project_memorymaze/memoryMaze/third_party/grpc/third_party/cares/cares/README.md")
set(CPACK_RESOURCE_FILE_WELCOME "/home/sbkim/miniconda3/envs/torchbeast/lib/python3.7/site-packages/cmake/data/share/cmake-3.26/Templates/CPack.GenericWelcome.txt")
set(CPACK_RPM_COMPONENT_INSTALL "1")
set(CPACK_RPM_COMPRESSION_TYPE "xz")
set(CPACK_RPM_Devel_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_Devel_PACKAGE_ARCHITECTURE "x86_64")
set(CPACK_RPM_Devel_PACKAGE_NAME "c-ares-devel")
set(CPACK_RPM_Devel_PACKAGE_REQUIRES "cmake >= 3.1.0, libcares >= 1.17.2")
set(CPACK_RPM_Devel_PACKAGE_SUMMARY "Development files for compiling against c-ares.")
set(CPACK_RPM_Library_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_Library_PACKAGE_ARCHITECTURE "x86_64")
set(CPACK_RPM_Library_PACKAGE_NAME "libcares")
set(CPACK_RPM_Library_PACKAGE_SUMMARY "The c-ares binary library.")
set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
set(CPACK_RPM_PACKAGE_AUTOPROV "1")
set(CPACK_RPM_PACKAGE_DESCRIPTION "A C library for asynchronous DNS requests")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_PACKAGE_RELEASE "1")
set(CPACK_RPM_PACKAGE_URL "https://c-ares.haxx.se/")
set(CPACK_RPM_Tools_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_Tools_PACKAGE_ARCHITECTURE "x86_64")
set(CPACK_RPM_Tools_PACKAGE_NAME "c-ares-tools")
set(CPACK_RPM_Tools_PACKAGE_REQUIRES "libcares >= 1.17.2")
set(CPACK_RPM_Tools_PACKAGE_SUMMARY "Tools for using c-ares.")
set(CPACK_SET_DESTDIR "OFF")
set(CPACK_SOURCE_GENERATOR "TBZ2;TGZ;TXZ;TZ")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/home/sbkim/project_memorymaze/memoryMaze/build/temp.linux-x86_64-cpython-37/CPackSourceConfig.cmake")
set(CPACK_SOURCE_RPM "OFF")
set(CPACK_SOURCE_TBZ2 "ON")
set(CPACK_SOURCE_TGZ "ON")
set(CPACK_SOURCE_TXZ "ON")
set(CPACK_SOURCE_TZ "ON")
set(CPACK_SOURCE_ZIP "OFF")
set(CPACK_SYSTEM_NAME "Linux")
set(CPACK_THREADS "1")
set(CPACK_TOPLEVEL_TAG "Linux")
set(CPACK_WIX_SIZEOF_VOID_P "8")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "/home/sbkim/project_memorymaze/memoryMaze/build/temp.linux-x86_64-cpython-37/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()
