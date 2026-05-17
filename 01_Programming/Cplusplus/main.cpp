#include <iostream>
#include <mysql/mysql.h>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib> // for system()

using namespace std;

// 定义记录结构体，用于存储日期、体重和身高

struct Record {
    string date;   // 日期
    float weight;  // 体重
    float height;  // 身高
};

// BMI记录管理类
class BMITracker {
private:
    MYSQL* conn;  // MySQL连接对象
    MYSQL_RES* res;  // 查询结果
    MYSQL_ROW row;   // 每行结果

public:
    // 构造函数，初始化数据库连接并创建表
    BMITracker(const string& db_host, const string& db_user, const string& db_pass, const string& db_name) {
        conn = mysql_init(0);
        if (conn == NULL) {
            cerr << "mysql_init() failed" << endl;
            exit(1);
        }

        conn = mysql_real_connect(conn, db_host.c_str(), db_user.c_str(), db_pass.c_str(), db_name.c_str(), 0, NULL, 0);
        if (conn == NULL) {
            cerr << "mysql_real_connect() failed: " << mysql_error(conn) << endl;
            exit(1);
        }

        // 创建表，用于存储记录
        string create_table = "CREATE TABLE IF NOT EXISTS records ("
                              "id INT AUTO_INCREMENT PRIMARY KEY, "
                              "date VARCHAR(20) NOT NULL, "
                              "weight FLOAT NOT NULL, "
                              "height FLOAT NOT NULL, "
                              "bmi FLOAT NOT NULL);";
        if (mysql_query(conn, create_table.c_str())) {
            cerr << "创建表失败: " << mysql_error(conn) << endl;
            exit(1);
        }
    }

    // 析构函数，关闭数据库连接
    ~BMITracker() {
        mysql_free_result(res);
        mysql_close(conn);
    }

    // 添加记录
    void addRecord(const string& date, float weight, float height) {
        // 计算BMI
        float bmi = weight / (height * height);
        // 插入记录的SQL语句
        string insert = "INSERT INTO records (date, weight, height, bmi) VALUES ('" + date + "', " +
                        to_string(weight) + ", " + to_string(height) + ", " + to_string(bmi) + ");";

        if (mysql_query(conn, insert.c_str())) {
            cerr << "添加记录失败: " << mysql_error(conn) << endl;
        } else {
            cout << "记录添加成功!" << endl;
        }
    }

    // 删除记录
    void deleteRecord(const string& date) {
        // 删除记录的SQL语句
        string del = "DELETE FROM records WHERE date = '" + date + "';";
        if (mysql_query(conn, del.c_str())) {
            cerr << "删除记录失败: " << mysql_error(conn) << endl;
        } else {
            cout << "记录删除成功!" << endl;
        }
    }

    // 显示所有记录
    void displayRecords() {
        // 查询所有记录的SQL语句
        string query = "SELECT date, weight, height, bmi FROM records;";
        if (mysql_query(conn, query.c_str())) {
            cerr << "查询记录失败: " << mysql_error(conn) << endl;
            return;
        }

        res = mysql_store_result(conn);
        if (res == NULL) {
            cerr << "获取查询结果失败: " << mysql_error(conn) << endl;
            return;
        }

        // 输出表头
        cout << "日期\t\t体重\t身高\tBMI" << endl;
        while ((row = mysql_fetch_row(res))) {
            string date = row[0];
            float weight = stof(row[1]);
            float height = stof(row[2]);
            float bmi = stof(row[3]);
            cout << date << "\t" << weight << "\t" << height << "\t" << bmi << endl;
        }
    }

    // 获取所有记录，用于绘制图表
    vector<Record> getRecords() {
        vector<Record> records;
        string query = "SELECT date, weight, height FROM records;";
        if (mysql_query(conn, query.c_str())) {
            cerr << "查询记录失败: " << mysql_error(conn) << endl;
            return records;
        }

        res = mysql_store_result(conn);
        if (res == NULL) {
            cerr << "获取查询结果失败: " << mysql_error(conn) << endl;
            return records;
        }

        while ((row = mysql_fetch_row(res))) {
            string date = row[0];
            float weight = stof(row[1]);
            float height = stof(row[2]);
            records.push_back({date, weight, height});
        }
        return records;
    }
};

// 使用 Gnuplot 绘制 BMI 变化趋势图
void plotBMI(const vector<Record>& records) {
    // 写入数据到文件
    ofstream data_file("bmi_data.txt");
    if (!data_file.is_open()) {
        cerr << "无法创建数据文件!" << endl;
        return;
    }

    for (const auto& record : records) {
        float bmi = record.weight / (record.height * record.height);
        data_file << record.date << " " << bmi << endl; // 每行写入日期和 BMI
    }
    data_file.close();

    // 写入 Gnuplot 脚本
    ofstream plot_script("plot_bmi.gp");
    if (!plot_script.is_open()) {
        cerr << "无法创建绘图脚本文件!" << endl;
        return;
    }

    plot_script << "set terminal png size 800,600\n";
    plot_script << "set output 'bmi_trend.png'\n";
    plot_script << "set title 'BMI 变化趋势'\n";
    plot_script << "set xlabel '日期'\n";
    plot_script << "set ylabel 'BMI'\n";
    plot_script << "set xdata time\n";
    plot_script << "set timefmt '%Y-%m-%d'\n";
    plot_script << "set format x '%Y-%m-%d'\n";
    plot_script << "set grid\n";
    plot_script << "plot 'bmi_data.txt' using 1:2 with linespoints title 'BMI'\n";
    plot_script.close();

    // 调用 Gnuplot 绘图
    system("gnuplot plot_bmi.gp");
    cout << "BMI 图表已生成: bmi_trend.png" << endl;
}

int main() {
    // 使用 MySQL 数据库连接参数
    string db_host = "localhost";
    string db_user = "root";
    string db_pass = "password";  // 替换为实际的密码
    string db_name = "bmi_tracker";

    BMITracker tracker(db_host, db_user, db_pass, db_name);  // 初始化数据库

    while (true) {
        // 菜单选项
        cout << "\n1. 添加记录\n2. 删除记录\n3. 显示记录\n4. 绘制BMI图表\n5. 退出\n";
        int choice;
        cin >> choice;

        if (choice == 1) {
            // 添加记录
            string date;
            float weight, height;

            cout << "请输入日期 (YYYY-MM-DD): ";
            cin >> date;
            cout << "请输入体重 (kg): ";
            cin >> weight;
            cout << "请输入身高 (m): ";
            cin >> height;

            tracker.addRecord(date, weight, height);
        } else if (choice == 2) {
            // 删除记录
            string date;
            cout << "请输入要删除的日期 (YYYY-MM-DD): ";
            cin >> date;

            tracker.deleteRecord(date);
        } else if (choice == 3) {
            // 显示记录
            tracker.displayRecords();
        } else if (choice == 4) {
            // 绘制 BMI 图表
            vector<Record> records = tracker.getRecords();
            plotBMI(records);
        } else if (choice == 5) {
            // 退出程序
            break;
        } else {
            cout << "无效选项，请重试。" << endl;
        }
    }

    return 0;
}
