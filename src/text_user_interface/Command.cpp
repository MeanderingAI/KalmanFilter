#include <Command.hpp>
#include <iostream>
#include <map>
#include <string>

std::map<std::string, Command> commandRegistry;

// Define your commands using the macro
REGISTER_COMMAND(hello, "Prints a greeting.", [](const std::vector<std::string>& args) {
    std::cout << "Hello, world!" << std::endl;
});

REGISTER_COMMAND(greet, "Greets a person by name.", [](const std::vector<std::string>& args) {
    if (args.size() > 0) {
        std::cout << "Hello, " << args[0] << "!" << std::endl;
    } else {
        std::cout << "Usage: greet <name>" << std::endl;
    }
});

REGISTER_COMMAND(help, "Lists all available commands.", [](const std::vector<std::string>& args) {
    std::cout << "Available commands:" << std::endl;
    for (const auto& pair : commandRegistry) {
        std::cout << "  " << pair.second.name << ": " << pair.second.description << std::endl;
    }
});

REGISTER_COMMAND(exit, "Exits the application.", [](const std::vector<std::string>& args) {
    // This lambda will simply print a message, the main loop handles the actual exit.
    std::cout << "Exiting..." << std::endl;
});