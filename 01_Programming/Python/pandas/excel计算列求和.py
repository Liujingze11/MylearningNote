import pandas as pd

# 改成你真实的文件名
df = pd.read_excel('text.xlsx', sheet_name='Sheet1')

# 获取第一列
values = df.iloc[:, 0]

# 累计和初始化
cumulative = []
running_total = 0

for val in values:
    try:
        num = float(val)
    except (ValueError, TypeError):
        num = 0
    running_total += num
    cumulative.append(running_total)

# 创建结果 DataFrame
result_df = pd.DataFrame({
    '原始值': values,
    '累计和': cumulative
})

# 写入 CSV
result_df.to_csv('sheet2_sum_result.csv', index=False, encoding='utf-8-sig')
print("✅ 成功：已生成 sheet2_sum_result.csv")
