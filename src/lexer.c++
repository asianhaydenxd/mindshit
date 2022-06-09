#include <iostream>
#include <string>
#include <stack>

using std::cout;
using std::string;
using std::stack;

struct Position {
    int index;
    int line;
    int column;
    string fileName;
    string fileText;

    Position next() {
        return {
            index + 1,
            fileText[index] == '\n' ? line + 1 : line,
            fileText[index] == '\n' ? 0 : column + 1,
            fileName,
            fileText,
        };
    }
};

inline Position startPosition(const string fileName, const string text) {
    return {0, 0, 0, fileName, text};
}

enum TokenType {
    KW, ID,
    INT, CHAR, STR,
    START, END,
};

struct Token {
    TokenType type;
    string value;
    string text;
    Position start;
};

struct Error {
    bool isError;
    string name;
    string info;
    Position start;
    Position end;
};

struct Lex {
    stack<Token> tokens;
    Error error;
};

Lex lex(const string fileName, const string text) {
    const string numbers = "1234567890";
    const string operators[] = {
        "+", "-", "/", "*", "%",
        "=", "+=", "-=", "*=", "/=", "%=",
        "++", "--",
        "==", ">", "<", ">=", "<=" "!=",
        "(", ")", "[", "]", "{", "}",
    };
    const string keywords[] = {
        "let"
    };

    Position position = startPosition(fileName, text);

    stack<Token> tokens;
    
    Error error;
    error.isError = false;
    
    tokens.push({START, "", "", position});

    while (tokens.top().type != END) {
        if (position.index > text.length()) {
            tokens.push({END, "", "", position});
        }
        else if (numbers.find(text[position.index]) != string::npos) {
            tokens.push({INT, "", "", position});
        }
        else {
            error.isError = true;
            error.name = "IllegalCharError";
            error.info = "stray '" + string(1, text[position.index]) + "' in program";
            
            return {tokens, error};
        }

        position = position.next();
    }

    return {tokens, error};
}

int main() {
    Lex test = lex("a.ms", "12\n34");
    cout << test.error.info << "\n";

    return 0;
}