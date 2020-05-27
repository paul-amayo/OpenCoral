project(coral_apps)

add_executable( coral_cpu_example coral_cpu_example.cpp)
target_link_libraries(coral_cpu_example open_coral)

add_executable( coral_gpu_example coral_gpu_example.cpp)
target_link_libraries(coral_gpu_example open_coral coral_cuda)

