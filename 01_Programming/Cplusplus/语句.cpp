#include <iostream>
#include <limits>

using namespace std;

// 定义活动选项的枚举类型，便于后续处理用户选择
enum Activity {
  BUY_TICKET = 1,  // 购买门票
  OPTION_2,        // 其他选项2（未开发）
  OPTION_3,        // 其他选项3（未开发）
  OPTION_4         // 其他选项4（未开发）
};

// 获取用户有效的活动选择
Activity getValidActivity() {
  int activity;
  while (true) {
    cout << "请输入你的选择: ";
    // 检查输入是否合法且在允许范围内（1~4）
    if (cin >> activity && (activity == BUY_TICKET || activity == OPTION_2 || activity == OPTION_3 || activity == OPTION_4)) {
      break; // 输入合法，跳出循环
    } else {
      cin.clear(); // 清除cin的错误标志
      cin.ignore(numeric_limits<streamsize>::max(), '\n'); // 忽略当前行的无效输入
      cout << "无效输入，请重新输入" << endl;
    }
  }
  // 将整数转换为对应的枚举类型并返回
  return static_cast<Activity>(activity);
}

// 获取用户输入的有效整数（带提示信息）
int getValidInput(const string& prompt, int& value) {
  while (!(cin >> value)) {
    cin.clear(); // 清除输入错误标志
    cin.ignore(numeric_limits<streamsize>::max(), '\n'); // 忽略错误输入
    cout << "无效输入，请重新输入" << prompt << ": ";
  }
  return value;
}

int main() {
  Activity activity; // 用户选择的活动类型
  int age;           // 用户年龄
  int money;         // 用户剩余资产

  // 欢迎界面与选项提示
  cout << "欢迎来到SINO乐园" << endl;
  cout << "如果购买门票请按1，查看其他选项请按2，查看其他选项请按3，查看其他选项请按4" << endl;

  // 获取用户的选择
  activity = getValidActivity();

  // 根据用户选择的活动执行对应逻辑
  switch (activity) {
    case BUY_TICKET:
      // 提示用户输入年龄
      cout << "请输入你的年龄: ";
      getValidInput("年龄", age);

      // 提示用户输入剩余资产
      cout << "请输入你的剩余资产: ";
      getValidInput("剩余资产", money);

      // 判断资产是否足够
      if (money >= 100) {
        // 判断年龄是否满18岁
        if (age < 18) {
          cout << "您未满18岁，不能购买门票" << endl;
        } else {
          cout << "购买成功" << endl;
        }
      } else {
        cout << "余额不足" << endl;
      }
      break;

    case OPTION_2:
      // 选项2占位
      cout << "未开发" << endl;
      break;

    case OPTION_3:
      // 选项3占位
      cout << "未开发" << endl;
      break;

    case OPTION_4:
      // 选项4占位
      cout << "未开发" << endl;
      break;

    default:
      // 理论上不会到达此处
      cout << "无效的选择" << endl;
      break;
  }

  return 0;
}