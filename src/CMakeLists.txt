add_library(tinympc STATIC
    consolidate.cu
)


target_include_directories(tinympc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..) 