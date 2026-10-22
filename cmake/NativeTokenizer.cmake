# Tokenization uses Hugging Face's native Rust implementation through a tiny C ABI.
find_program(VAE_CARGO cargo REQUIRED)
set(VAE_TOKENIZER_ROOT "${CMAKE_SOURCE_DIR}/source/ml/tokenizer")
set(VAE_TOKENIZER_TARGET "${CMAKE_BINARY_DIR}/native-tokenizer")
if(WIN32)
    set(VAE_TOKENIZER_LIB "${VAE_TOKENIZER_TARGET}/release/vae_tokenizer.lib")
else()
    set(VAE_TOKENIZER_LIB "${VAE_TOKENIZER_TARGET}/release/libvae_tokenizer.a")
endif()
if(APPLE)
    set(VAE_TOKENIZER_ENV "MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()
add_custom_command(OUTPUT "${VAE_TOKENIZER_LIB}"
    COMMAND "${CMAKE_COMMAND}" -E env ${VAE_TOKENIZER_ENV} "${VAE_CARGO}" build --release --locked --manifest-path "${VAE_TOKENIZER_ROOT}/Cargo.toml"
        --target-dir "${VAE_TOKENIZER_TARGET}"
    DEPENDS "${VAE_TOKENIZER_ROOT}/Cargo.toml" "${VAE_TOKENIZER_ROOT}/Cargo.lock" "${VAE_TOKENIZER_ROOT}/src/lib.rs"
    VERBATIM)
add_custom_target(vae_tokenizer_build DEPENDS "${VAE_TOKENIZER_LIB}")
add_library(vae_tokenizer STATIC IMPORTED GLOBAL)
set_target_properties(vae_tokenizer PROPERTIES IMPORTED_LOCATION "${VAE_TOKENIZER_LIB}")
add_dependencies(vae_tokenizer vae_tokenizer_build)
if(WIN32)
    # A Rust staticlib does not carry its native system-library dependencies
    # into the C++ link. Propagate them to every tokenizer consumer.
    target_link_libraries(vae_tokenizer INTERFACE
        kernel32 ntdll userenv ws2_32 dbghelp bcrypt advapi32)
elseif(APPLE)
    target_link_libraries(vae_tokenizer INTERFACE "-framework Security" "-framework SystemConfiguration")
elseif(UNIX)
    target_link_libraries(vae_tokenizer INTERFACE pthread dl m)
endif()
