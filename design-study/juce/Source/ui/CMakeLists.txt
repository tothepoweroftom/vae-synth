# Add this directory from a JUCE application or plug-in build. No study assets or model runtime.
add_library(pitcher_study_ui INTERFACE)
target_sources(pitcher_study_ui INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/PitcherControls.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../OrbitView.cpp)
target_include_directories(pitcher_study_ui INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(pitcher_study_ui INTERFACE juce::juce_gui_basics juce::juce_audio_utils)
