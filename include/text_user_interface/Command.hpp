#ifndef COMMAND_H
#define COMMAND_H

#include <string>
#include <functional>
#include <vector>
#include <map>

// A struct to hold command data
struct Command {
    std::string name;
    std::string description;
    std::function<void(const std::vector<std::string>&)> action;
};

// Global registry for commands
extern std::map<std::string, Command> commandRegistry;

// Macro for easy command registration
#define REGISTER_COMMAND(name, desc, code) \
    namespace { \
        struct Command_##name { \
            Command_##name() { \
                commandRegistry[#name] = {#name, desc, code}; \
            } \
        }; \
        static Command_##name registrar_##name; \
    }
    
#endif // COMMAND_H