# include "iostream"
using namespace std;
// 符号常量：#define 名称（标识符） 常量值，符号常量定义在代码的头部
// 符号常量的定义 不需要分号结尾
#define MAX 100


int main()
{
  // 基础语法学习：这是学习程序的基础
  // 首先学习cout，直接复制：
  // cout << "在双引号中加上显示的内容"<< endl;
  // 并修改显示的内容
  int n1;
  int n2;
  cout << "请输入一个整数："<< endl;
  // 学习输入
  cout << "请输入一个整数："<< endl;
  cin >> n1;
  cout << n1 << endl;
  cout << "请输入一个字符："<< endl;
  cin >> n2;
  cout << n2 << endl;
  unsigned int num3 = 30;
  cout << num3 << endl;
  // 1. 字面常量：整型、实型、字符、字符串

  // 这是整型，也就是整数
  21;
  // 这是实型（小数）
  180.5;
  // 这是字符，‘’进行包围，单个字符

  'c';  //''里面只能是1个字母，不能是0或者更多
  // 字符串，“”进行包围，任意个字符
  ""; // 空字符串，里面是0个字符，是OK的
  "C";
  "ssss";

  // 通过cout将这些内容全部打印到控制台上
  cout<<21<<endl;
  cout<<180.5<<endl;
  cout<<'c'<<endl;
  cout<<"ssss"<<endl;

  // 2.
  cout<<"MAX"<<endl;

  // 3.变量
  // 变量的声明（定义），变量类型 变量名；
  int age; //整型的变量
  //变量的赋值
  age = 1;
  //变量的使用（取值），直接使用变量名称即可
  cout << "年龄为:" << age << endl;
  //变量的变化
  age = 2;
  cout << "年龄增长一岁后为：" << age << endl;
  //变量声明与赋值同时进行。
  float height=180.1;
  cout << "身高为:" << height << endl;
  //一次性声明多个变量（并赋值）

  // 4. 数据类型
  // 4.1 整型
  int a;

  signed int c;
  string b;


  // 4.2 实型
  // float单精度浮点数，4字节，6～7位的有效位数
  float nu1 = 123456789; //只提供前7位的准确输出
  float nu2 = 1.23456789;
  cout << nu1 << endl;
  cout << nu2 << endl;

  // double双精度浮点数，8字节，15～16

  // long double


  cout << fixed; //设置小数显示
  cout << nu1 << endl;
  cout << nu2 << endl;

  //4.3 常量类型的确定

  // 不带有后缀的数字的类型确定，整数的类型最小是int，小数的类型最小是double
  cout << "10L的字节数是:" << sizeof(10L) << endl;
  cout << "10L的字节数是:" << sizeof(10L) << endl;
  cout << "10L的字节数是:" << sizeof(10L) << endl;
  cout << "10L的字节数是:" << sizeof(10L) << endl;

  //4.4 字符型

  char ch = 65;
  cout << ch << endl;

  char ch2 = 'a';
  cout << ch2+1 << endl;

  char ch3 = 'a' + 1;

  // 4.5 转义字符
  // 转义字符：将普通的字符使用\
  // \t制表符演示，效果等同于键盘的tab键，一个\t可以补充到8个字符位



  //4.6 字符串的不同风格
  // c语言风格的字符串
  char s1[] = "hello";
  char *s2 = "hello";

  string s = "hello"; // C++ string类型的字符串

  // 字符串的拼接
  string name = "小黑";
  string major = "物理";

  // 4.7 布尔型
  // 布尔：bool 字面量仅仅有2个：true或false
  bool flag = false;  // true 标识符号是真，
  bool flag2 = true;
  cout << flag << endl;

  //5. 运算符
  //算术运算符
  //单目（只有1个操作数）操作符：
  int num1 = 5+5; //
  int num2 = 5-5; //
  cout << "5+5 = " << num1 << endl;
  cout << "5-5 = " << num2 << endl;

  // 比较运算符


  // C语言风格字符串，直接应用

  // 三元运算符：表达式 ？v1:v2
  int sanyuan;
}