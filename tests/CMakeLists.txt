set(NOETHER_TESTS
    test_smoke
    test_force
    test_integrators
    test_config
)

foreach(t ${NOETHER_TESTS})
    add_executable(${t} ${t}.cpp)
    target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR})
    target_compile_options(${t} PRIVATE -O3 -Wall -Wextra)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# CUDA test: built only when CUDA is available; skips (77) without a device.
if(NOETHER_HAS_CUDA)
    add_executable(test_cuda test_cuda.cpp ${CMAKE_SOURCE_DIR}/core/cuda/kernels.cu)
    target_include_directories(test_cuda PRIVATE ${CMAKE_SOURCE_DIR})
    target_compile_options(test_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
    )
    target_compile_definitions(test_cuda PRIVATE NOETHER_USE_CUDA)
    target_link_libraries(test_cuda PRIVATE CUDA::cudart)
    add_test(NAME test_cuda COMMAND test_cuda)
    set_tests_properties(test_cuda PROPERTIES SKIP_RETURN_CODE 77)
endif()
