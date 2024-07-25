# add_executable(BasicModel BasicModel.cu)
# target_link_libraries(BasicModel LINK_PUBLIC tinympc ${CUSPARSE_LIBRARY})


# set_target_properties(BasicModel PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )

# add_executable(StreamModul StreamModul.cu)
# target_link_libraries(StreamModul LINK_PUBLIC tinympc ${CUSPARSE_LIBRARY})


# set_target_properties(StreamModul PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )

# add_executable(MoreFancy MoreFancy.cu)
# target_link_libraries(MoreFancy LINK_PUBLIC tinympc ${CUSPARSE_LIBRARY})
# set_target_properties(MoreFancy PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )


add_executable(Combined Combined.cu)
target_link_libraries(Combined LINK_PUBLIC tinympc ${CUSPARSE_LIBRARY})
set_target_properties(Combined PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# add_executable(FancyStream FancyStream.cu)
# target_link_libraries(FancyStream LINK_PUBLIC tinympc ${CUSPARSE_LIBRARY})
# set_target_properties(FancyStream PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )



# add_executable(StreamCompared StreamCompared.cu)
# target_link_libraries(StreamCompared LINK_PUBLIC tinympc ${CUSPARSE_LIBRARY})
# set_target_properties(StreamCompared PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )




