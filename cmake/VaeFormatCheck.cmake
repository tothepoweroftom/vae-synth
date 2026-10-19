# ctest vae_format_check: fails when any C++ file under source/ or tests/ is
# not formatted per .clang-format (written against clang-format 18; other
# major versions may disagree on a few lines).
#   cmake -D CLANG_FORMAT=<exe> -D ROOT=<repo> -P VaeFormatCheck.cmake

file(GLOB_RECURSE files
    "${ROOT}/source/*.h" "${ROOT}/source/*.cpp"
    "${ROOT}/tests/*.h" "${ROOT}/tests/*.cpp")

execute_process(
    COMMAND "${CLANG_FORMAT}" --dry-run --Werror --style=file ${files}
    WORKING_DIRECTORY "${ROOT}"
    RESULT_VARIABLE result)

if(NOT result EQUAL 0)
    message(FATAL_ERROR "Unformatted C++ (see above). Fix with: clang-format -i $(git ls-files 'source/*.h' 'source/*.cpp' 'tests/*.h' 'tests/*.cpp')")
endif()
