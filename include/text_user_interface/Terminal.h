#ifndef TERMINAL_H
#define TERMINAL_H

#include <iostream>
#include <string>

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
    #ifndef _WIN32
    termios original_tty;
    #endif
    
    void setRawMode();
    void restoreMode();
};

#endif // TERMINAL_H