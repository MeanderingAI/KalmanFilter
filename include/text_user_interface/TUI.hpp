#ifndef TUI_H
#define TUI_H

#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#include <cstdio>
#endif

namespace TUI {
    // Platform-independent functions
    void setCursorPosition(int row, int col);
    void clearScreen();
    void setRawMode();
    void restoreMode();
    char getch_unbuffered();

    // Platform-specific implementations
    #ifdef _WIN32
    void setCursorPosition(int row, int col) {
        COORD coord;
        coord.X = col - 1;
        coord.Y = row - 1;
        SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), coord);
    }
    void clearScreen() {
        system("cls");
    }
    void setRawMode() {
        // Not necessary for _getch()
    }
    void restoreMode() {
        // Not necessary for _getch()
    }
    char getch_unbuffered() {
        return _getch();
    }
    #else
    void setCursorPosition(int row, int col) {
        std::cout << "\x1B[" << row << ";" << col << "H";
    }
    void clearScreen() {
        std::cout << "\x1B[2J\x1B[H";
    }
    termios original_tty;
    void setRawMode() {
        termios tty;
        tcgetattr(STDIN_FILENO, &tty);
        original_tty = tty;
        tty.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &tty);
    }
    void restoreMode() {
        tcsetattr(STDIN_FILENO, TCSANOW, &original_tty);
    }
    char getch_unbuffered() {
        char ch;
        std::cin.get(ch);
        return ch;
    }
    #endif
}

#endif // TUI_H