# vae_p0_eval: renders p0 Find / typical / word-list starts to WAV and profiles Find under load
# (docs/PROMPT-EVAL.md, docs/JOINT-VAE-LAB.md). Experimental; not part of the plug-in.
juce_add_console_app(vae_p0_eval PRODUCT_NAME "vae_p0_eval" VERSION 0.1.0)
target_sources(vae_p0_eval PRIVATE P0EvalMain.cpp ../compare/Features.cpp)
target_compile_definitions(vae_p0_eval PRIVATE JUCE_MODAL_LOOPS_PERMITTED=1 JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
target_link_libraries(vae_p0_eval PRIVATE vae_engine vae_joint_experiment juce::juce_audio_formats juce::juce_dsp
    juce::juce_recommended_config_flags juce::juce_recommended_warning_flags)
vae_copy_onnxruntime_next_to(vae_p0_eval)
if(WIN32)
    target_link_libraries(vae_p0_eval PRIVATE psapi)
endif()
