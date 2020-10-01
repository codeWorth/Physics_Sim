cmake_minimum_required(VERSION 3.10)
project(PhysicsSim)

add_executable(phys "src/main.cpp" "src/glad.c")
target_include_directories(phys PRIVATE "include/")
target_link_directories(phys PRIVATE "lib/lib-mingw-w64/")
target_link_libraries(phys glfw3)

target_compile_options(phys PUBLIC -mavx2 -O3 -march=native)
install(TARGETS phys DESTINATION bin)