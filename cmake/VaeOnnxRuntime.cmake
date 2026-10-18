# ONNX Runtime from Microsoft's prebuilt release archives.
#
# The plug-in does not link against onnxruntime: it compiles against the C/C++
# headers with ORT_API_MANUAL_INIT and loads the shared library at runtime
# (source/ml/OrtLoader.cpp). That keeps the plug-in loadable when the library
# is missing (ML features just report "unavailable"), avoids the Windows
# "DLL next to the host .exe" problem, and lets us ship the library inside
# each plug-in bundle.
#
# Provides:
#   vae::onnxruntime_headers   INTERFACE target with include dirs
#   VAE_ORT_RUNTIME_FILE       the shared library to bundle
#   VAE_ORT_RUNTIME_NAME       its file name (what OrtLoader looks for)
#   vae_bundle_onnxruntime(<juce plugin target>)

set(VAE_ORT_VERSION "1.22.0" CACHE STRING "ONNX Runtime release to download")
set(VAE_ORT_ROOT "" CACHE PATH "Existing extracted ONNX Runtime package (empty = download)")

if(NOT VAE_ORT_ROOT)
    if(WIN32)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64|aarch64")
            set(_ort_pkg "onnxruntime-win-arm64-${VAE_ORT_VERSION}")
        else()
            set(_ort_pkg "onnxruntime-win-x64-${VAE_ORT_VERSION}")
        endif()
        set(_ort_ext "zip")
    elseif(APPLE)
        set(_ort_pkg "onnxruntime-osx-universal2-${VAE_ORT_VERSION}")
        set(_ort_ext "tgz")
    else()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(_ort_pkg "onnxruntime-linux-aarch64-${VAE_ORT_VERSION}")
        else()
            set(_ort_pkg "onnxruntime-linux-x64-${VAE_ORT_VERSION}")
        endif()
        set(_ort_ext "tgz")
    endif()

    include(FetchContent)
    FetchContent_Declare(onnxruntime_prebuilt
        URL https://github.com/microsoft/onnxruntime/releases/download/v${VAE_ORT_VERSION}/${_ort_pkg}.${_ort_ext}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(onnxruntime_prebuilt)
    set(VAE_ORT_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}")
endif()

if(NOT EXISTS "${VAE_ORT_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime headers not found under ${VAE_ORT_ROOT}/include")
endif()

if(WIN32)
    set(VAE_ORT_RUNTIME_NAME "onnxruntime.dll")
    set(VAE_ORT_RUNTIME_FILE "${VAE_ORT_ROOT}/lib/onnxruntime.dll")
elseif(APPLE)
    set(VAE_ORT_RUNTIME_NAME "libonnxruntime.dylib")
    file(GLOB _ort_libs "${VAE_ORT_ROOT}/lib/libonnxruntime.*.dylib")
    list(GET _ort_libs 0 VAE_ORT_RUNTIME_FILE)
else()
    set(VAE_ORT_RUNTIME_NAME "libonnxruntime.so")
    set(VAE_ORT_RUNTIME_FILE "${VAE_ORT_ROOT}/lib/libonnxruntime.so.${VAE_ORT_VERSION}")
endif()

if(NOT EXISTS "${VAE_ORT_RUNTIME_FILE}")
    message(FATAL_ERROR "ONNX Runtime library not found: ${VAE_ORT_RUNTIME_FILE}")
endif()
message(STATUS "ONNX Runtime ${VAE_ORT_VERSION}: ${VAE_ORT_RUNTIME_FILE}")

add_library(vae_onnxruntime_headers INTERFACE)
add_library(vae::onnxruntime_headers ALIAS vae_onnxruntime_headers)
target_include_directories(vae_onnxruntime_headers SYSTEM INTERFACE "${VAE_ORT_ROOT}/include")
target_compile_definitions(vae_onnxruntime_headers INTERFACE
    ORT_API_MANUAL_INIT=1
    VAE_ORT_RUNTIME_NAME="${VAE_ORT_RUNTIME_NAME}")

# Copy the runtime next to each built plug-in format (macOS: Contents/Frameworks).
function(vae_bundle_onnxruntime plugin_target)
    foreach(format VST3 AU Standalone CLAP)
        set(t "${plugin_target}_${format}")
        if(NOT TARGET ${t})
            continue()
        endif()
        if(APPLE)
            set(dest "$<TARGET_BUNDLE_CONTENT_DIR:${t}>/Frameworks")
        else()
            set(dest "$<TARGET_FILE_DIR:${t}>")
        endif()
        add_custom_command(TARGET ${t} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${dest}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VAE_ORT_RUNTIME_FILE}" "${dest}/${VAE_ORT_RUNTIME_NAME}"
            VERBATIM)
    endforeach()
endfunction()

# Copy the runtime next to a plain executable (tests, tools).
function(vae_copy_onnxruntime_next_to exe_target)
    add_custom_command(TARGET ${exe_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VAE_ORT_RUNTIME_FILE}"
                "$<TARGET_FILE_DIR:${exe_target}>/${VAE_ORT_RUNTIME_NAME}"
        VERBATIM)
endfunction()
