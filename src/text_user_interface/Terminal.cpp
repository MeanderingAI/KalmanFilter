#include "Terminal.h"

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


std::string Terminal::readLine() {
    std::string input;
    char ch;
    while ( (ch = getch()) != '\r' && ch != '\n') {
        if (ch == 127 || ch == 8) { // Backspace
            if (!input.empty()) {
                input.pop_back();
                std::cout << "\b \b" << std::flush;
            }
        } else if (ch >= 32 && ch <= 126) { // Printable characters
            input += ch;
            std::cout << ch << std::flush;
        }
    }
    return input;
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