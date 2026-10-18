# vae_params: parameter store (docs/specs/parameter-manager.md).
# Expects JUCE to have been added (add_subdirectory(<JUCE>)) by the including project.

add_library(vae_params STATIC
    ParamSpec.h
    ParamSpec.cpp
    LinkRule.h
    ParamStore.h
    ParamStore.cpp
    Attachments.h
    Attachments.cpp)

target_compile_features(vae_params PUBLIC cxx_std_20)

# Users include "params/ParamStore.h".
target_include_directories(vae_params PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

# JUCE modules are INTERFACE libraries carrying their sources; linking them PUBLIC gives
# consumers the include paths, module definitions and module sources they need.
target_link_libraries(vae_params
    PUBLIC
        juce::juce_audio_processors
        juce::juce_gui_basics
    PRIVATE
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)

target_compile_definitions(vae_params
    PUBLIC
        JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
    PRIVATE
        # Kept PRIVATE: juce_add_console_app / juce_add_gui_app / juce_add_plugin define their
        # own JUCE_STANDALONE_APPLICATION value, and a PUBLIC =0 would clash (macro redefined).
        JUCE_STANDALONE_APPLICATION=0)
