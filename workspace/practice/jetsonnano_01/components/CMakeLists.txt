add_library(all_components INTERFACE)

file(GLOB subdirectories LIST_DIRECTORIES true *)

foreach(component_path ${subdirectories})
    if(IS_DIRECTORY ${component_path})
        get_filename_component(component_name ${component_path} NAME)
        add_subdirectory(${component_name})

        target_link_libraries(all_components INTERFACE ${component_name})
    endif()
endforeach()
