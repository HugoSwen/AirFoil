import pandas as pd
import os

# 定义数据目录
data_dir = 'data'

# 需要拆分的文件列表
files_to_split = ['x.csv', 'z.csv']

# 检查数据目录是否存在
if not os.path.isdir(data_dir):
    print(f"错误：目录 '{data_dir}' 不存在。")
else:
    # 遍历文件列表
    for file_name in files_to_split:
        input_file = os.path.join(data_dir, file_name)

        # 检查文件是否存在
        if not os.path.isfile(input_file):
            print(f"警告：文件 '{input_file}' 未找到，已跳过。")
            continue

        print(f"正在处理文件: {input_file}")
        # 读取CSV文件，假设没有表头
        df = pd.read_csv(input_file, header=None)

        # 检查列数是否符合预期
        if df.shape[1] != 399:
            print(f"警告：文件 '{file_name}' 的列数不是399，已跳过。")
            continue

        # 前199列是下表面
        lower_surface_df = df.iloc[:, :199]

        # 后200列是上表面
        upper_surface_df = df.iloc[:, 199:]

        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(file_name)[0]

        # 定义输出文件路径
        lower_output_file = os.path.join(data_dir, f'{base_name}_lower.csv')
        upper_output_file = os.path.join(data_dir, f'{base_name}_upper.csv')

        # 将拆分后的数据保存到新的CSV文件，不包含索引和表头
        lower_surface_df.to_csv(lower_output_file, index=False, header=False)
        upper_surface_df.to_csv(upper_output_file, index=False, header=False)

        print(f"文件 '{file_name}' 已成功拆分为 '{os.path.basename(lower_output_file)}' 和 '{os.path.basename(upper_output_file)}'")

print("\n所有文件处理完毕。")
