cmake_minimum_required(VERSION 3.8)
project(multi_sensor_calibration)

find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_auto_find_build_dependencies()

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/multi-sensor-calibration
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  README.md
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_pytest_test(
    test_event_windows
    test/test_event_windows.py
  )
  ament_add_pytest_test(
    test_time_sync
    test/test_time_sync.py
  )
  ament_add_pytest_test(
    test_transforms
    test/test_transforms.py
  )
endif()

ament_auto_package(
  INSTALL_TO_SHARE
    config
)
