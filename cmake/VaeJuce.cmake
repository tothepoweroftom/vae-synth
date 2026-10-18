# JUCE: use a local checkout when VAE_JUCE_DIR is set, otherwise fetch the
# pinned tag once into the build tree.

set(VAE_JUCE_DIR "" CACHE PATH "Path to an existing JUCE checkout (empty = fetch)")
set(VAE_JUCE_TAG "8.0.10" CACHE STRING "JUCE tag fetched when VAE_JUCE_DIR is empty")

if(VAE_JUCE_DIR)
    add_subdirectory("${VAE_JUCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/juce-build" EXCLUDE_FROM_ALL)
else()
    include(FetchContent)
    FetchContent_Declare(juce
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG        ${VAE_JUCE_TAG}
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(juce)
endif()
