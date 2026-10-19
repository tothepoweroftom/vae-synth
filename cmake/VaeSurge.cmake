# Optional Surge XT DSP backend (GPL-3.0-or-later).
#
# Builds only Surge's headless `surge-common` library (no Surge JUCE, GUI or
# plug-in targets). Enabling VAE_WITH_SURGE makes the resulting plug-in
# binary a combined work under the GPL; see docs/LICENSING.md.
#
# VAE_SURGE_DIR: existing Surge checkout with submodules (empty = fetch the pin).

set(VAE_SURGE_DIR "" CACHE PATH "Path to a Surge XT checkout with submodules (empty = fetch)")
set(VAE_SURGE_REV "45568c7d60be52aa2a4f27dd7b7a074f56bd00ba" CACHE STRING "Surge XT commit fetched when VAE_SURGE_DIR is empty")

if(NOT VAE_SURGE_DIR)
    include(FetchContent)
    FetchContent_Declare(surge
        GIT_REPOSITORY https://github.com/surge-synthesizer/surge.git
        GIT_TAG        ${VAE_SURGE_REV}
        GIT_SUBMODULES
            libs/sst/sst-basic-blocks libs/sst/sst-cmake libs/sst/sst-cpputils
            libs/sst/sst-effects libs/sst/sst-filters libs/sst/sst-plugininfra
            libs/sst/sst-waveshapers libs/eurorack/eurorack libs/fmt
            libs/luajitlib/LuaJIT libs/luajitlib/cmake libs/oddsound-mts/MTS-ESP
            libs/pffft libs/simde libs/tuning-library libs/r8brain-free-src
            libs/zstd libs/PEGTL
        SOURCE_SUBDIR  _vae_do_not_configure)   # populate only; added below
    FetchContent_MakeAvailable(surge)
    set(VAE_SURGE_DIR "${surge_SOURCE_DIR}")
endif()

# Headless only: no Surge JUCE, plug-ins, test runner or CLAP.
set(SURGE_SKIP_JUCE_FOR_RACK ON CACHE BOOL "" FORCE)
set(SURGE_BUILD_TESTRUNNER OFF CACHE BOOL "" FORCE)
set(SURGE_BUILD_XT OFF CACHE BOOL "" FORCE)
set(SURGE_BUILD_FX OFF CACHE BOOL "" FORCE)
set(SURGE_BUILD_CLAP OFF CACHE BOOL "" FORCE)
set(SURGE_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)

add_subdirectory("${VAE_SURGE_DIR}" "${CMAKE_BINARY_DIR}/_deps/surge-build" EXCLUDE_FROM_ALL)

# Surge's common/CMakeLists.txt assumes it is the top-level project and refers
# to ${CMAKE_SOURCE_DIR}/libs/r8brain-free-src. Re-point those entries at the
# Surge tree instead of patching Surge.
# Plain string replacement: the paths may contain regex characters (C++, ...).
foreach(prop SOURCES INCLUDE_DIRECTORIES)
    get_target_property(values surge-common ${prop})
    set(fixed "")
    foreach(value IN LISTS values)
        string(REPLACE "${CMAKE_SOURCE_DIR}/libs/" "${VAE_SURGE_DIR}/libs/" value "${value}")
        list(APPEND fixed "${value}")
    endforeach()
    set_target_properties(surge-common PROPERTIES ${prop} "${fixed}")
endforeach()

# Surge's headers need SSE4.1 on x86 (Surge sets it directory-wide for itself).
add_library(vae_surge INTERFACE)
add_library(vae::surge ALIAS vae_surge)
target_link_libraries(vae_surge INTERFACE surge::surge-common)
target_compile_definitions(vae_surge INTERFACE VAE_WITH_SURGE=1)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86" AND NOT MSVC)
    target_compile_options(vae_surge INTERFACE -msse4.1)
endif()
