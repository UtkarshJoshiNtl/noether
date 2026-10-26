set(NOETHER_TESTS
    test_smoke
    test_force
    test_integrators
)

foreach(t ${NOETHER_TESTS})
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} PRIVATE noether_core)
    target_compile_options(${t} PRIVATE -O3 -Wall -Wextra)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
