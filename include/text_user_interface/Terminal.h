#ifndef TERMINAL_H
#define TERMINAL_H

#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

class Terminal {
public:
    Terminal();
    ~Terminal();

    void clearScreen();
    void setCursorPosition(int row, int col);
    char getch();
    void print(const std::string& text);
    std::string readLine();
    void clearToEnd();

private:
    std::vector<std::string> commandHistory;
    int historyIndex;

    #ifndef _WIN32
    termios original_tty;
    #endif
    
    void setRawMode();
    void restoreMode();
    void clearCurrentLine(); // Helper function
    void reprintLine(const std::string& line);
    std::string findMatchingCommand(const std::string& prefix);
    void drawPromptAndHint(const std::string& input, const std::string& hint);  
};

#endif // TERMINAL_H