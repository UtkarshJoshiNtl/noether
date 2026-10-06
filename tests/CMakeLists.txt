set(NOETHER_TESTS
    test_smoke
)

foreach(t ${NOETHER_TESTS})
    add_executable(${t} ${t}.cpp)
    target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR})
    target_compile_options(${t} PRIVATE -O3 -Wall -Wextra)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
