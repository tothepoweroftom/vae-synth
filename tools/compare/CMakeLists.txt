# vae_compare: renders presets on the original Surge patch, the Surge backend
# and the Lite backend and writes audio features + distances (docs/COMPARE.md).
# Only built with VAE_WITH_SURGE; not part of the plug-in.

juce_add_console_app(vae_compare PRODUCT_NAME "vae_compare" VERSION 0.1.0)

target_sources(vae_compare PRIVATE
    CompareMain.cpp
    Features.cpp
    SurgeOriginal.cpp)

target_compile_definitions(vae_compare PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)

target_link_libraries(vae_compare PRIVATE
    vae_engine
    juce::juce_audio_formats
    juce::juce_dsp
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)
