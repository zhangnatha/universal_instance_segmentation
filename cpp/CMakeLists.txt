cmake_minimum_required(VERSION 3.16...3.28)
project(universal_instance_seg_cpp LANGUAGES CXX)

# 将编译生成的可执行二进制文件统一输出至指定的 CMake 构建目录（例如 cpp/build）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# 构建选项配置
# 1. UBUNTU18_COMPAT：使 Linux 二进制兼容旧版 glibc 2.27（Ubuntu 18.04+），通过重定向 __libc_start_main 符号实现
# 2. BUILD_GPU：除默认 CPU 目标外，同时编译支持 CUDA 加速的 GPU 推理目标（带 _cuda 后缀）
option(UBUNTU18_COMPAT "Keep Linux binaries compatible with glibc 2.27" ON)
option(BUILD_GPU "Build CUDA-enabled targets in addition to CPU targets" OFF)

# ONNX Runtime 根目录与 CUDA/cuDNN 运行时库路径配置
set(ONNXRUNTIME_ROOT "" CACHE PATH "CPU ONNX Runtime package root")
set(ONNXRUNTIME_GPU_ROOT "" CACHE PATH "GPU ONNX Runtime package root")
set(ONNXRUNTIME_GPU_RUNTIME_ROOT "${CMAKE_CURRENT_LIST_DIR}/3rdparty/cuda11-runtime" CACHE PATH
  "CUDA/cuDNN runtime directory used by GPU ONNX Runtime")
set(ORT_ROOT "" CACHE PATH "Alias for ONNXRUNTIME_ROOT")
if(NOT ONNXRUNTIME_ROOT AND ORT_ROOT)
  set(ONNXRUNTIME_ROOT "${ORT_ROOT}")
endif()

# 自动推导或兜底解析 ONNX Runtime CPU 根目录
if(NOT ONNXRUNTIME_ROOT)
  if(WIN32)
    file(GLOB _ort_win_cands LIST_DIRECTORIES true "${CMAKE_CURRENT_LIST_DIR}/3rdparty/onnxruntime-win-x64*")
    list(APPEND _ort_win_cands "${CMAKE_CURRENT_LIST_DIR}/3rdparty/onnxruntime")
    foreach(_cand IN LISTS _ort_win_cands)
      if(EXISTS "${_cand}/include/onnxruntime_cxx_api.h")
        set(ONNXRUNTIME_ROOT "${_cand}")
        break()
      endif()
    endforeach()
  else()
    file(GLOB _ort_linux_cands LIST_DIRECTORIES true "${CMAKE_CURRENT_LIST_DIR}/3rdparty/onnxruntime-linux-x64*")
    list(APPEND _ort_linux_cands "${CMAKE_CURRENT_LIST_DIR}/3rdparty/onnxruntime")
    foreach(_cand IN LISTS _ort_linux_cands)
      if(EXISTS "${_cand}/include/onnxruntime_cxx_api.h")
        set(ONNXRUNTIME_ROOT "${_cand}")
        break()
      endif()
    endforeach()
  endif()
elseif(NOT IS_ABSOLUTE "${ONNXRUNTIME_ROOT}")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    get_filename_component(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/${ONNXRUNTIME_ROOT}" ABSOLUTE)
  elseif(EXISTS "${CMAKE_SOURCE_DIR}/../${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    get_filename_component(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/../${ONNXRUNTIME_ROOT}" ABSOLUTE)
  endif()
endif()

# 自动推导或兜底解析 OpenCV 路径
if(NOT OpenCV_DIR)
  foreach(_cand
      "${CMAKE_CURRENT_LIST_DIR}/3rdparty/opencv-install-4.5.5/lib/cmake/opencv4"
      "${CMAKE_CURRENT_LIST_DIR}/3rdparty/opencv/lib/cmake/opencv4"
      "${CMAKE_CURRENT_LIST_DIR}/3rdparty/opencv/build")
    if(EXISTS "${_cand}/OpenCVConfig.cmake")
      set(OpenCV_DIR "${_cand}")
      break()
    endif()
  endforeach()
elseif(NOT IS_ABSOLUTE "${OpenCV_DIR}")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${OpenCV_DIR}/OpenCVConfig.cmake")
    get_filename_component(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${OpenCV_DIR}" ABSOLUTE)
  elseif(EXISTS "${CMAKE_SOURCE_DIR}/../${OpenCV_DIR}/OpenCVConfig.cmake")
    get_filename_component(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../${OpenCV_DIR}" ABSOLUTE)
  endif()
endif()

# 查找 OpenCV 必需组件（核心矩阵计算、图像处理、图像编解码）
find_package(OpenCV REQUIRED CONFIG COMPONENTS core imgproc imgcodecs)

# 校验 ONNX Runtime CPU 根目录是否提供
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR
    "ONNXRUNTIME_ROOT is required. Run cpp/scripts/fetch_onnxruntime.sh (Linux) "
    "or cpp/scripts/fetch_onnxruntime.ps1 (Windows), or provide an installed ONNX Runtime package.")
endif()

# 配置指定目标的 ONNX Runtime 与 OpenCV 依赖、RPATH、编译器选项及系统兼容层
function(configure_onnx_target target_name ort_root)
  if(NOT EXISTS "${ort_root}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
      "ONNX Runtime headers not found for ${target_name}: ${ort_root}")
  endif()

  set(ort_library_var "_${target_name}_ort_library")
  # 清理可能残留的缓存变量，确保在同一构建目录中切换 CPU/GPU 配置时能重新定位对应动态库
  unset(${ort_library_var} CACHE)
  unset(${ort_library_var})
  find_library(${ort_library_var} NAMES onnxruntime
    PATHS "${ort_root}/lib" "${ort_root}/bin" NO_DEFAULT_PATH)
  if(NOT ${ort_library_var})
    message(FATAL_ERROR
      "ONNX Runtime shared library not found for ${target_name}: ${ort_root}")
  endif()

  target_include_directories(${target_name} PRIVATE
    "${ort_root}/include" ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${target_name} PRIVATE
    "${${ort_library_var}}" ${OpenCV_LIBS})

  # Linux 动态链接路径（RPATH）配置：
  # GPU 版 ONNX Runtime 在运行时通过 dlopen 加载 CUDA 执行提供者。
  # 此处将配套的 CUDA 11 / cuDNN 8 运行库路径注入 RPATH，使程序无需全局修改 LD_LIBRARY_PATH 即可直接运行。
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(target_rpath "${ort_root}/lib")
    if(ARGC GREATER 2 AND EXISTS "${ARGV2}/lib")
      list(APPEND target_rpath "${ARGV2}/lib")
      # 保留相对路径支持，确保打包部署到新机器上时二进制文件可自定位同级依赖库
      list(APPEND target_rpath "$ORIGIN/lib;$ORIGIN/../lib;$ORIGIN/../3rdparty/cuda11-runtime/lib")
    endif()
    set_target_properties(${target_name} PROPERTIES
      BUILD_RPATH "${target_rpath}"
      INSTALL_RPATH "$ORIGIN/lib;$ORIGIN/../lib;$ORIGIN/../runtime/lib;$ORIGIN/../3rdparty/cuda11-runtime/lib")
    if(ARGC GREATER 2 AND NOT MSVC)
      # 二进制中的 DT_RUNPATH 不会被通过 dlopen 延迟加载的动态库继承；
      # 通过禁用 new-dtags 使用传统 DT_RPATH，确保 CUDA provider 可以顺利解析 CUDA 11/cuDNN 8 依赖库。
      set_property(TARGET ${target_name} APPEND_STRING PROPERTY
        LINK_FLAGS " -Wl,--disable-new-dtags")
    endif()
  endif()

  # 编译器选项与宏定义配置（跨平台适配 MSVC 与 GCC/Clang）
  if(MSVC)
    target_compile_definitions(${target_name} PRIVATE
      NOMINMAX _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  # Ubuntu 18.04+ (glibc 2.27) 兼容性适配层：
  # 将 __libc_start_main 重定向绑定至旧版 GLIBC_2.2.5 符号，解决高版本系统编译后在低版本系统上缺少高版本 glibc 的问题
  if(UBUNTU18_COMPAT AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/common/glibc_compat.cpp")
      target_sources(${target_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/common/glibc_compat.cpp")
    elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/models/detectron2_maskrcnn/src/glibc_compat.cpp")
      target_sources(${target_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/models/detectron2_maskrcnn/src/glibc_compat.cpp")
    endif()
    set_property(TARGET ${target_name} APPEND_STRING PROPERTY
      LINK_FLAGS " -Wl,--wrap=__libc_start_main")
  endif()

  # 旧版本 GCC（如 Ubuntu 18.04 原生 GCC 7.5/8.x）在引入 <filesystem> 时需要显式链接 stdc++fs
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(${target_name} PRIVATE stdc++fs)
  endif()
endfunction()

# ------------------------------------------------------------------------------
# 目标 1：Detectron2 Mask R-CNN 实例分割推理可执行程序
# 输入：image [3, H, W] -> 输出：boxes [N, 4], scores [N], classes [N], mask_probs [N, 1, MH, MW]
# ------------------------------------------------------------------------------
add_executable(detectron2_maskrcnn_infer
  models/detectron2_maskrcnn/src/main.cpp)
target_compile_features(detectron2_maskrcnn_infer PRIVATE cxx_std_17)
configure_onnx_target(detectron2_maskrcnn_infer "${ONNXRUNTIME_ROOT}")

# ------------------------------------------------------------------------------
# 目标 2：Ultralytics YOLO-seg (YOLOv8/11/26-seg) 实例分割推理可执行程序
# 输入：image [1, 3, H, W] -> 输出：Rank-3 预测分支与 Rank-4 原型掩码分支 (Prototypes)
# ------------------------------------------------------------------------------
add_executable(ultralytics_yolo_infer
  models/ultralytics_yolo/src/main.cpp)
target_compile_features(ultralytics_yolo_infer PRIVATE cxx_std_17)
configure_onnx_target(ultralytics_yolo_infer "${ONNXRUNTIME_ROOT}")

# ------------------------------------------------------------------------------
# 目标 3：RF-DETR 实例分割推理可执行程序
# 输入：image [1, 3, H, W] -> 输出：dets [1, Q, 4], labels [1, Q, C+1], 可选 masks [1, Q, MH, MW]
# ------------------------------------------------------------------------------
add_executable(rfdetr_infer
  models/rfdetr/src/main.cpp)
target_compile_features(rfdetr_infer PRIVATE cxx_std_17)
configure_onnx_target(rfdetr_infer "${ONNXRUNTIME_ROOT}")

# ------------------------------------------------------------------------------
# 可选 GPU 目标配置（启用 BUILD_GPU 时生成对应 *_cuda 推理目标）
# ------------------------------------------------------------------------------
if(BUILD_GPU)
  if(NOT ONNXRUNTIME_GPU_ROOT)
    file(GLOB _ort_gpu_cands LIST_DIRECTORIES true "${CMAKE_CURRENT_LIST_DIR}/3rdparty/onnxruntime-cuda*")
    foreach(_cand IN LISTS _ort_gpu_cands)
      if(EXISTS "${_cand}/include/onnxruntime_cxx_api.h")
        set(ONNXRUNTIME_GPU_ROOT "${_cand}")
        break()
      endif()
    endforeach()
  endif()
  if(NOT ONNXRUNTIME_GPU_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_GPU_ROOT is required when BUILD_GPU=ON")
  endif()

  # 跨平台 CUDA/cuDNN 运行时库校验
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(NOT EXISTS "${ONNXRUNTIME_GPU_RUNTIME_ROOT}/lib/libcublasLt.so.11" OR
       NOT EXISTS "${ONNXRUNTIME_GPU_RUNTIME_ROOT}/lib/libcudnn.so.8")
      message(FATAL_ERROR
        "CUDA 11 runtime is incomplete: expected libcublasLt.so.11 and "
        "libcudnn.so.8 under ${ONNXRUNTIME_GPU_RUNTIME_ROOT}/lib. "
        "Install the matching CUDA 11/cuDNN 8 runtime there; do not symlink "
        "CUDA 12 libraries to CUDA 11 SONAMEs.")
    endif()
  elseif(WIN32)
    # Windows 环境下如果指定了运行时路径，校验关键 DLL 是否存在
    if(EXISTS "${ONNXRUNTIME_GPU_RUNTIME_ROOT}/bin")
      set(_cuda_dll_dir "${ONNXRUNTIME_GPU_RUNTIME_ROOT}/bin")
    else()
      set(_cuda_dll_dir "${ONNXRUNTIME_GPU_RUNTIME_ROOT}/lib")
    endif()
  endif()

  foreach(model_target IN ITEMS detectron2_maskrcnn ultralytics_yolo rfdetr)
    add_executable(${model_target}_infer_cuda
      models/${model_target}/src/main.cpp)
    target_compile_features(${model_target}_infer_cuda PRIVATE cxx_std_17)
    target_compile_definitions(${model_target}_infer_cuda PRIVATE ENABLE_CUDA_PROVIDER=1)
    configure_onnx_target(${model_target}_infer_cuda "${ONNXRUNTIME_GPU_ROOT}"
      "${ONNXRUNTIME_GPU_RUNTIME_ROOT}")
  endforeach()
endif()

# ------------------------------------------------------------------------------
# C++ 可执行程序一键打包命令：
# 执行 'cmake --build <build_dir> --target package_infer' 即可自动收集二进制程序及其所有
# 动态链接库（ONNX Runtime / OpenCV / CUDA 运行库）、运行脚本与说明文档，输出独立免安装部署包。
# ------------------------------------------------------------------------------
if(WIN32)
  add_custom_target(package_infer
    COMMAND powershell.exe -NoProfile -ExecutionPolicy Bypass -File
            "${CMAKE_CURRENT_LIST_DIR}/scripts/package_windows.ps1"
            -BuildDir "${CMAKE_BINARY_DIR}" -Force
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    COMMENT "Packaging Windows C++ inference executables and runtime libraries..."
  )
else()
  add_custom_target(package_infer
    COMMAND bash "${CMAKE_CURRENT_LIST_DIR}/scripts/package_linux.sh"
            --build-dir "${CMAKE_BINARY_DIR}" --force
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    COMMENT "Packaging Linux C++ inference executables and runtime libraries..."
  )
endif()
add_custom_target(package_all DEPENDS package_infer)
