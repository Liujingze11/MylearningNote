#include <iostream>  // 包含输入输出功能（cout）
#include <string>    // 包含 string 类型支持
using namespace std; // 使用 std 命名空间，方便使用 cout, string 等

// 定义一个模板类 Box，T 是一个占位符类型（generic type）
template <typename T>
class Box {
private:
    T data;  // 成员变量，类型由用户指定，例如 int, string 等
public:
    // 设置数据的方法，参数是 T 类型的常量引用
    void set(const T& val) {
        data = val;
    }

    // 获取数据的方法，返回类型是 T，函数为常量函数，不修改类状态
    T get() const {
        return data;
    }
};

int main() {
    // 创建一个存储 int 类型的 Box 对象
    Box<int> intBox;
    intBox.set(123);  // 设置值为 123
    cout << intBox.get() << endl;  // 输出：123

    // 创建一个存储 string 类型的 Box 对象
    Box<string> strBox;
    strBox.set("模板类");  // 设置值为字符串“模板类”
    cout << strBox.get() << endl;  // 输出：模板类

    return 0;
}

