set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -dc")


add_library(tinympc STATIC
    consolidate.cu
    Kernel.cu
    HelpFunction.cu
    FancyDesign.cu
    GancyDesign.cu
    )


target_include_directories(tinympc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..) 