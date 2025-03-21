import pandas as pd
import os
import argparse
import json

parser = argparse.ArgumentParser(description="Data Analysis")
parser.add_argument('--model_name', type=str, default="gpt-4o", help='The name of the model')
args = parser.parse_args()

MODEL_NAME = args.model_name
MAX_TURN = 7

def save_csv(df, file_path):
    # 使用pandas的to_csv函数将DataFrame保存到CSV文件
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
                try:
                    data = json.load(file)
                except:
                    print(file_path)
                # 确保读取的内容是列表，并将其扩展到all_data中
                if isinstance(data, list):
                    all_data.extend(data)
                elif isinstance(data, dict):
                    all_data.append(data)
    return all_data

def process_one_turn(INSTITUTE):
    # 整理路径
    porject_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 获取当前文件目录
    PATH = f"{porject_dir}/dataset" #设置所有这个任务要保存的根路径
    # 输出文件的地址
    ITERATION_PATH = f"{PATH}/outputs/IOR-Dynamic/{INSTITUTE}/{MODEL_NAME}"
    
    OUTPUT_PATH = f"{ITERATION_PATH}/{INSTITUTE}_ior_{MAX_TURN}.csv"
    JSON_PATH = f"{ITERATION_PATH}/{INSTITUTE}_ior_{MAX_TURN}.json"
    
    # 如果交互时中间出现问题中断交互，但又已经在目录下保存了很多json临时文件了，则注释上一行，执行下面这行
    result_list = read_all_json_files_to_list(ITERATION_PATH)
    # 下面整理文件，如果所有逻辑顺利，则提取所有json文件进行合并，并删除所有单独的json文件
    # 保存所有json结果到一个文件
    save_json(result_list, JSON_PATH)
        
    # 把json转化为dataframe
    final_df = pd.DataFrame(result_list)
    
     # 保存数据
    save_csv(final_df, OUTPUT_PATH)
    
    # 整理log文件
    log_result = f"\n\n结果保存在：{OUTPUT_PATH}\n"
    log_result += f"Multi-query的结果：\n"
    log_result += f"Probe的结果：\n"

    for i in range(MAX_TURN):
        title = f"sucess_{i}"
        log_result += f"probe-{i}结果：{final_df[title].sum()}/{len(result_list)}：{final_df[title].sum()/len(result_list)}\n"
    
    # 保存日志
    with open(f"{ITERATION_PATH}/ior_result.txt", 'a', encoding='utf-8') as f:
        f.write(log_result)

def main():
    INSTITUTES = ["hospital-1","hospital-2"]

    for INSTITUTE in INSTITUTES:
        process_one_turn(INSTITUTE) 


if __name__ == "__main__":
    main()