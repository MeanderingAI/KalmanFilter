#include <text_user_interface/Terminal.h>
#include <text_user_interface/Command.hpp>
#include <map>

extern std::map<std::string, Command> commandRegistry;

// Class constructor
Terminal::Terminal() {
    setRawMode();
    #ifndef _WIN32
    std::cout << "\x1B[?25l";
    #endif
}

// Class destructor
Terminal::~Terminal() {
    restoreMode();
    #ifndef _WIN32
    std::cout << "\x1B[?25h";
    #endif
}

void Terminal::clearScreen() {
    #ifdef _WIN32
    system("cls");
    #else
    std::cout << "\x1B[2J\x1B[H";
    #endif
}

void Terminal::clearCurrentLine() {
    std::cout << "\r\x1B[K" << std::flush;
}

void Terminal::reprintLine(const std::string& line) {
    std::cout << "\r" << line << std::flush;
}

void Terminal::setCursorPosition(int row, int col) {
    #ifdef _WIN32
    COORD coord;
    coord.X = col - 1;
    coord.Y = row - 1;
    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), coord);
    #else
    std::cout << "\x1B[" << row << ";" << col << "H";
    #endif
}

char Terminal::getch() {
    #ifdef _WIN32
    return _getch();
    #else
    return getchar();
    #endif
}

void Terminal::print(const std::string& text) {
    std::cout << text;
}


std::string Terminal::findMatchingCommand(const std::string& input) {
    if (input.empty()) {
        return "";
    }

    // Iterate through all registered commands to find a match.
    for (const auto& pair : commandRegistry) {
        const std::string& commandName = pair.first;
        
        // Check if the current command name starts with the user's input.
        // We use string::rfind to check if the input is found at the beginning of the command name.
        if (commandName.rfind(input, 0) == 0) {
            // Found a match, return the rest of the command name as a hint.
            return commandName;
        }
    }
    
    // No match found.
    return "";
}

void Terminal::drawPromptAndHint(const std::string& input, const std::string& hint) {
    // Save the cursor position
    std::cout << "\x1B[s" << std::flush;
    
    // Move to the beginning of the line and clear it
    std::cout << "\r\x1B[K" << std::flush;

    // Print the prompt and user's input
    std::cout << "> " << input;

    // Print the hint in a dimmed color (ANSI escape code \x1B[2m)
    // The hint is the full command minus the part the user has already typed.
    if (!hint.empty()) {
        std::string hintPart = hint.substr(input.size());
        std::cout << "\x1B[2m" << hintPart << "\x1B[0m" << std::flush;
    }

    // Restore the cursor position to the end of the user's input
    std::cout << "\x1B[u" << std::flush;
}

std::string Terminal::readLine() {
    std::string currentInput;
    historyIndex = commandHistory.size(); // Start at the end of history
    
    // Draw the initial prompt.
    std::cout << "> " << std::flush;

    while (true) {
        char ch = getch();

        if (ch == '\t') { // Tab key
            std::string hint = findMatchingCommand(currentInput);
            if (!hint.empty()) {
                currentInput = hint;
                drawPromptAndHint(currentInput, ""); // Redraw without a hint
            }
        } else if (ch == '\r' || ch == '\n') { // Enter key
            std::cout << std::endl;
            if (!currentInput.empty()) {
                commandHistory.push_back(currentInput);
            }
            return currentInput;
        } else if (ch == 127 || ch == 8) { // Backspace
            if (!currentInput.empty()) {
                currentInput.pop_back();
                std::string hint = findMatchingCommand(currentInput);
                drawPromptAndHint(currentInput, hint);
            }
        } else if (ch == '\x1B') { // Escape sequence for arrow keys
            getch(); // Skip '['
            char key = getch();

            if (key == 'A') { // Up arrow
                if (historyIndex > 0) {
                    historyIndex--;
                    currentInput = commandHistory[historyIndex];
                    drawPromptAndHint(currentInput, "");
                }
            } else if (key == 'B') { // Down arrow
                if (historyIndex < commandHistory.size() - 1) {
                    historyIndex++;
                    currentInput = commandHistory[historyIndex];
                    drawPromptAndHint(currentInput, "");
                } else if (historyIndex == commandHistory.size() - 1) {
                    historyIndex++;
                    currentInput = "";
                    drawPromptAndHint(currentInput, "");
                }
            }
        } else if (ch >= 32 && ch <= 126) { // Printable characters
            currentInput += ch;
            std::string hint = findMatchingCommand(currentInput);
            drawPromptAndHint(currentInput, hint);
        }
    }
}

void Terminal::clearToEnd() {
    #ifdef _WIN32
    // On Windows, this is more complex, so a simple newline may be enough for this example.
    // For a real solution, you'd use the Win32 API to clear the line.
    std::cout << "\n";
    #else
    std::cout << "\x1B[K" << std::flush;
    #endif
}

void Terminal::setRawMode() {
    #ifndef _WIN32
    termios tty;
    tcgetattr(STDIN_FILENO, &tty);
    original_tty = tty;
    tty.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &tty);
    #endif
}

void Terminal::restoreMode() {
    #ifndef _WIN32
    tcsetattr(STDIN_FILENO, TCSANOW, &original_tty);
    #endif
}