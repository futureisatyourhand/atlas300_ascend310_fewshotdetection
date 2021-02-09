
#ifndef COMMAND_LINE_H_
#define COMMAND_LINE_H_
#include <gflags/gflags.h>
#include <string>

/// @brief Define flag for showing help message <br>
static const char help_message[] = "Print a usage message.";
DEFINE_bool(h, false, help_message);

static const char i_message[] = "Optional. Specify the input image, default: ../data/test.jpg";
DEFINE_string(i, "../data/test.jpg", i_message);

/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "main [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                             " << help_message << std::endl;
    std::cout << "    -i '<path>'                    " << i_message << std::endl;
}

#endif  // COMMAND_LINE_H_
