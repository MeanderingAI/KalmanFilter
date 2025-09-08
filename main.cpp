#include <text_user_interface/Terminal.h>
#include <text_user_interface/Command.hpp>
#include <sstream>
#include <iterator>
#include <iostream>
#include <vector>
#include <string>


int main() {
    Terminal term;
    term.clearScreen();
    term.setCursorPosition(1, 1);
    term.print("Welcome to the TUI. Type 'help' to get started.");

    while (true) {
        //term.setCursorPosition(3, 1);
        //term.print(">");
        //term.setCursorPosition(3, 3);

        std::string line = term.readLine();

        std::istringstream iss(line);
        std::vector<std::string> parts{std::istream_iterator<std::string>{iss},
                                       std::istream_iterator<std::string>{}};

        if (parts.empty()) continue;

        term.setCursorPosition(5, 1); // Move to a new line for command output
        term.clearToEnd(); // Clear the rest of the line

        std::string commandName = parts[0];
        parts.erase(parts.begin());

        if (commandName == "quit") {
            break;
        }


        // --- Clear the screen after command is submitted ---
        term.clearScreen();

        term.print("\n\r");
        if (commandRegistry.count(commandName)) {
            // Print the command that was run
            term.print(commandName + "\n");
            commandRegistry[commandName].action(parts);
        } else {
            term.print("Unknown command: " + commandName);
        }
    }

    return 0;
}