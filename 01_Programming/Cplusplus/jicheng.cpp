#include <iostream>
#include <string>
using namespace std;

// 父类：Animal（动物类）
class Animal {
private:
    string name;  // 封装：名字是私有的，不能外部访问

public:
    // 构造函数：初始化名字
    Animal(string n) {
        name = n;
    }

    // 提供接口：获取名字
    string getName() {
        return name;
    }

    // 虚函数：动物发出叫声（允许子类重写）
    virtual void sound() {
        cout << name << " 发出了未知的声音..." << endl;
    }
};

// 子类：Dog（狗）
class Dog : public Animal {  // public继承
public:
    Dog(string n) : Animal(n) {}  // 调用父类构造函数初始化名字

    // 重写虚函数 sound()
    void sound() override {
        cout << getName() << " 汪汪汪！" << endl;
    }
};

// 子类：Cat（猫）
class Cat : public Animal {
public:
    Cat(string n) : Animal(n) {}

    void sound() override {
        cout << getName() << " 喵喵喵！" << endl;
    }
};

// 主函数
int main() {
    // 创建对象
    Dog dog("小狗");
    Cat cat("小猫");

    // 父类指针实现多态
    Animal* animal;

    // 指向dog
    animal = &dog;
    animal->sound();  // 调用 Dog 的 sound()

    // 指向cat
    animal = &cat;
    animal->sound();  // 调用 Cat 的 sound()

    return 0;
}
