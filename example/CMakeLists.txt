add_executable(test test.cu)


target_link_libraries(test LINK_PUBLIC tinympc ${CUSPARSE_LIBRARY})