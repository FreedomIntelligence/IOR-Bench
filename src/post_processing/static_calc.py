import pandas as pd
import os
import argparse
import json
 
parser = argparse.ArgumentParser(description="Data Analysis")
parser.add_argument('--model_name', type=str, default="gpt-4o", help='The name of the model')

args = parser.parse_args()
MODEL_NAME = args.model_name

def save_csv(df, file_path):
    df.to_csv(file_path, index=False)
    # 输出保存成功的信息
    print(f'DataFrame已成功保存到 {file_path}')

def save_json(role_dict, file_path):
    # 将结果保存到单独的JSON文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(role_dict, f, ensure_ascii=False, indent=4)

def read_all_json_files_to_list(directory):
    all_data = []
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # 确保读取的内容是列表，并将其扩展到all_data中
                if isinstance(data, list):
                    all_data.extend(data)
                elif isinstance(data, dict):
                    all_data.append(data)
    return all_data

def process_counts(df):
    # 统计不同项及每项的数量
    count_stats = df.groupby('ask_count').size().reset_index(name='total_count')

    # 统计每组中 sucess_count 等于 1 的数量
    success_stats = df[df['sucess_count'] == 1].groupby('ask_count').size().reset_index(name='sucess_count_1')

    # 合并两个统计结果
    final_stats = pd.merge(count_stats, success_stats, on='ask_count', how='left').fillna(0)

    # 将 sucess_count_1 列转换为整数类型
    final_stats['sucess_count_1'] = final_stats['sucess_count_1'].astype(int)

    # 计算 sucess_count_1 占 total_count 的比例
    final_stats['success_ratio'] = final_stats['sucess_count_1'] / final_stats['total_count']

    # 初始化 logtext 字符串并添加标题行
    logtext = "\n"
    # 通过循环将 DataFrame 的每一行添加到 logtext 中
    for index, row in final_stats.iterrows():
        logtext += f"{str(int(row['ask_count']))} 轮成功率为{str(int(row['sucess_count_1']))} / {str(int(row['total_count']))}\n"

    return logtext

def main(TASK_TYPE):
    
    # 整理路径
    porject_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 获取当前文件目录
    PATH = f"{porject_dir}/dataset" #设置所有这个任务要保存的根路径
    
    # 输出文件的地址
    ITERATION_PATH = f"{PATH}/outputs/IOR-Static/{MODEL_NAME}/{TASK_TYPE}"
    
    # 输出文件的目录
    OUTPUT_PATH = f"{ITERATION_PATH}/{TASK_TYPE}_triage.csv"
    JSON_PATH = f"{ITERATION_PATH}/{TASK_TYPE}_triage.json"
    
    # 读取目录下的每个临时文件
    result_list = read_all_json_files_to_list(ITERATION_PATH)
    
    # 保存所有json结果到一个文件
    save_json(result_list, JSON_PATH)
        
    # # 把json转化为dataframe
    final_df = pd.DataFrame(result_list)
    #  # 保存数据
    save_csv(final_df, OUTPUT_PATH)
    # breakpoint()
    # 整理log文件
    count_ones = final_df["success_count"].sum()
    log_result = f"结果保存在：{ITERATION_PATH}\n"
    log_result += f"成功预测科室数: {count_ones}/{len(result_list)}\n"
    accuracy = count_ones / len(result_list)
    log_result += f"准确率为: {accuracy}\n"
    
    # 保存日志
    with open(f"{ITERATION_PATH}/triage_result.txt", 'a', encoding='utf-8') as f:
        f.write(log_result)

if __name__ == "__main__":
    TASK_TYPE_LIST = ["cot", "zero_shot", "few_shot", "majority vote"]
    for TASK_TYPE in TASK_TYPE_LIST:
        main(TASK_TYPE)