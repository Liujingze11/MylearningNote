#include <iostream>
using namespace std;

// 父类：Parent
class Parent {
public:
    int moneyPublic = 100;
protected:
    int moneyProtected = 200;
private:
    int moneyPrivate = 300;  // 不管哪种继承，永远无法继承

public:
    void showParent() {
        cout << "Parent内部访问："
             << moneyPublic << ", "
             << moneyProtected << ", "
             << moneyPrivate << endl;
    }
};

// 子类1：public继承
class Child_Public : public Parent {
public:
    void show() {
        cout << "Child_Public访问：" << endl;
        cout << "moneyPublic = " << moneyPublic << endl;       // ✅ 能访问
        cout << "moneyProtected = " << moneyProtected << endl; // ✅ 能访问
        // cout << moneyPrivate << endl;  // ❌ 无法访问
    }
};

// 子类2：protected继承
class Child_Protected : protected Parent {
public:
    void show() {
        cout << "Child_Protected访问：" << endl;
        cout << "moneyPublic = " << moneyPublic << endl;       // ✅ 能访问（变成protected）
        cout << "moneyProtected = " << moneyProtected << endl; // ✅ 能访问
        // cout << moneyPrivate << endl;  // ❌ 无法访问
    }
};

// 子类3：private继承
class Child_Private : private Parent {
public:
    void show() {
        cout << "Child_Private访问：" << endl;
        cout << "moneyPublic = " << moneyPublic << endl;       // ✅ 能访问（变成private）
        cout << "moneyProtected = " << moneyProtected << endl; // ✅ 能访问
        // cout << moneyPrivate << endl;  // ❌ 无法访问
    }
};

int main() {
    Child_Public c1;
    c1.show();
    cout << "外部访问 Child_Public:" << endl;
    cout << c1.moneyPublic << endl;  // ✅ public继承还能访问
    // cout << c1.moneyProtected << endl;  // ❌ 外部访问不了protected

    cout << "======================" << endl;

    Child_Protected c2;
    c2.show();
    // cout << c2.moneyPublic << endl;  // ❌ 外部访问不了，已变protected

    cout << "======================" << endl;

    Child_Private c3;
    c3.show();
    // cout << c3.moneyPublic << endl;  // ❌ 外部访问不了，已变private

    return 0;
}
