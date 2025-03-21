import pandas as pd
import os
import re
import argparse
import json
import ast
from utils.simulator import ChatBot
from multiprocessing import Pool
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Data Analysis")
parser.add_argument('--model_name', type=str, default="gpt-4o", help='The name of the model')
parser.add_argument('--run_type', type=str, default="normal", help='parallel or normal')
parser.add_argument('--processes_num', type=int, default=4, help='number of max process')
args = parser.parse_args()
MODEL_NAME = args.model_name
PROBE = False #因为dynamic evaluation是采取探针的形式，计算每轮对话的准确率，所以如果想对static evaluation做同样的统计，则把PROBE的值改为True，结果文件里会统计探针结果
PROSS_NUM = args.processes_num
RUN_TYPE = args.run_type
STRATEGIES = ["cot", "zero_shot", "few_shot", "majority vote"]
PROMPT_STRATEGIES = STRATEGIES

def read_csv(file):
    df = pd.read_csv(file)
    return df

def df_to_dict(df):
    # 将每一行转换成字典，并存储在一个列表中
    dict_list = df.to_dict('records')
    return dict_list

def save_json(role_dict, file_path):
    # 将结果保存到单独的JSON文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(role_dict, f, ensure_ascii=False, indent=4)

def zero_shot_prompt(department, dialog_history):    
    instruction = f'''
    你是医院分诊台的导诊医生，负责给病人推荐合适的科室。你负责的科室是：
    {str(department)}。
    根据到当前为止对话历史中病人的信息，你要从医院的分诊科室表中为他推荐一个科室。
    注意：
    1.你只按要求对当前轮的对话做回应，绝不能补全后面好几轮的对话！！
    2.你只能从分诊科室表中推荐一个最合适的科室，不要推荐多个科室。
    '''
    prompt = f'''
    ### 新的对话
    {dialog_history}
    推荐科室:
    '''
    return instruction, prompt

def few_shot_prompt(department, dialog_history):
    instruction = f'''
    你是医院分诊台的导诊医生，负责给病人推荐合适的科室。你负责的科室是：
    {str(department)}。
    根据到当前为止对话历史中病人的信息，你要从医院的分诊科室表中为他推荐一个科室。
    注意：
    1.你只按要求对当前轮的对话做回应，绝不能补全后面好几轮的对话！！
    2.你只能从分诊科室表中推荐一个最合适的科室，不要推荐多个科室。
    3. 不要回答任何与推荐科室无关的内容，不要诊断。
    4. 用中文回答。
    5. 下面是一些示例对话，请参照示例对话，根据给定的对话内容推荐最合适的科室。

    ### 示例 1
    对话:
    病人: 医生，我最近总是感到疲倦，而且情绪低落。
    医生: 医生: 这种情况持续多久了？有没有其他不适？
    病人: 有好几个月了，睡不好，食欲差，也提不起精神。
    推荐科室: 临床心理门诊

    ### 示例 2
    对话:
    病人: 我孩子这几天一直发烧，还有点咳嗽。
    医生: 孩子发烧持续了几天了？体温有多高？
    病人: 已经三天了，体温在38.5度左右，咳嗽时有痰。
    推荐科室: 门儿科

    ### 示例 3
    对话:
    病人: 我最近经常咳嗽，还有点喘不过气来。
    医生: 咳嗽有多久了？是干咳还是有痰？喘不过气的时候是在活动后还是休息时也有？
    病人: 大概两周了，主要是干咳，最近感觉一活动就喘不过气。
    推荐科室: 呼吸与危重症医学科
    '''
    
    prompt = f'''
    ### 新的对话
    {dialog_history}
    推荐科室:
    '''
    return instruction, prompt

def cot_prompt(department, dialog_history):

    instruction = f'''
    你是医院分诊台的导诊医生，负责给病人推荐合适的科室。你负责的科室是：
    {str(department)}。
    根据到当前为止对话历史中病人的信息，你要从医院的分诊科室表中为他推荐一个科室。
    注意：
    1.你只按要求对当前轮的对话做回应，绝不能补全后面好几轮的对话！！
    2.你只能从分诊科室表中推荐一个最合适的科室，不要推荐多个科室。
    3. 输出内容应该包括：推荐理由、结论两部分。示例格式为： #### 推荐理由\n...\n#### 结论\n...
    '''
    prompt = f'''
    对话记录：
    {dialog_history}
    #### 推荐理由:
    '''
    
    return instruction, prompt

def ior(department, dialog_history):
    if TASK_TYPE == "few_shot":
        ior_instrution, prompt = few_shot_prompt(department, dialog_history)
        return ior_instrution, prompt
    if TASK_TYPE == "cot":
        ior_instrution, prompt = cot_prompt(department, dialog_history)
        return ior_instrution, prompt
    else: # for zero shot and majority vote strategies
        ior_instrution, prompt = zero_shot_prompt(department, dialog_history)
        return ior_instrution, prompt

def detect_results(item_list, target_string):
    found_items = [item for item in item_list if item in target_string]
    if len(found_items) == 0:
        return ""
    else:
        return found_items[0]

def process_dialogue(input_str):
    input_str = input_str.replace("'", '"')
    dialogues = json.loads(input_str)
    # 格式化输出
    formatted_dialogue = ""
    for dialogue in dialogues:
        for role, sentence in dialogue.items():
            formatted_dialogue += f"{role}: {sentence}\n"       
    return formatted_dialogue

def remove_last_doctor_reply(dialogue: str) -> str:
    # 将对话按换行符分割为列表
    lines = dialogue.strip().split('\n')
    # 逆序遍历对话行，找到最后一个医生的回复
    for i in range(len(lines)-1, -1, -1):
        if lines[i].startswith('医生:'):
            # 去掉最后一个医生的回复并返回
            return '\n'.join(lines[:i])
    # 如果没有找到医生的回复，返回原对话
    return dialogue

def process_dialog_content(content):
    # 将字符串转换为列表
    list_data = ast.literal_eval(content)
    # 删除列表中的空值
    cleaned_list = [item for item in list_data if len(item) != 0]
    return cleaned_list

def interact_probe(iorRobot, ior_instruction, dialouge_history, turns, department):
    probe_list = {}
    curr_dialogue = f""
    count = 0
    for turn in turns:
        curr_dialogue += f"病人: {turn['病人']}\n"
        response = iorRobot.call(ior_instruction, dialouge_history)
        curr_dialogue += f"医生: {turn['医生']}\n"
        key_name = f"turn_{count}"
        probe_list[key_name] = detect_results(department, response)
        count += 1
    return probe_list

def majority_vote(iorRobot, ior_instruction, prompt, department):
    voting_count = 5 # how many times for voting
    votes = []
    for v in range(voting_count):
        vote = iorRobot.call(ior_instruction, prompt)
        vote_extract = detect_results(department, vote)
        votes.append(vote_extract)
    # 创建一个字典来存储每个字符串的出现次数
    frequency = {}
    for string in votes:
        if string in frequency:
            frequency[string] += 1
        else:
            frequency[string] = 1
    best_vote = max(frequency, key=frequency.get)
    return best_vote

def interact_prompting(iorRobot, ior_instruction, prompt, department, TASK_TYPE):
    if "majority" in TASK_TYPE:
        response = majority_vote(iorRobot, ior_instruction, prompt, department)
    else:
        response = iorRobot.call(ior_instruction, prompt)
    return response

def process_role(iorRobot, role, ITERATION_PATH, department, TASK_TYPE):
    dialouge_history = remove_last_doctor_reply(process_dialogue(role["对话内容"])) #提取历史对话记录
    ior_instruction, prompt = ior(department, dialouge_history)
    if PROBE: 
        '''
        用探针方法，计算每轮对话的结果
        '''
        turns = process_dialog_content(role["对话内容"])
        probe_results = interact_probe(iorRobot, ior_instruction, dialouge_history, turns, department)
        role.update(probe_results) # 把探针的结果加到role_dict中
        for r in range(len(probe_results)):
            key_name = f"sucess_{r}"
            role[key_name] =  1 if role["lable"] in role[f"turn_{r}"] else 0
    else:
        '''
        不用探针方法，直接计算整段对话的导诊结果
        '''
        
        response = interact_prompting(iorRobot, ior_instruction, prompt, department, TASK_TYPE)
        role["history"] = dialouge_history
        role["ior_result"] = response
        role["ior_result_extracted"] = detect_results(department, response)
        role["success_count"] = 1 if role["lable"] in role["ior_result_extracted"] else 0
    # 把当前一轮的结果存到json临时文件中
    role_path = f"{ITERATION_PATH}/role_{role['id']}.json"
    save_json(role, role_path)
    print(role_path)

def count_files(directory_path, DATASET_NUM):
    # 列表用来存储提取的数字
    numbers = []
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        # 使用正则表达式查找文件名中的数字
        match = re.search(r'\d+', filename)
        if match:
            # 提取并存储数字
            numbers.append(int(match.group()))
    # 找出0到500之间的所有数字
    all_numbers = set(range(int(DATASET_NUM)))
    # 找出numbers列表中没有的数字
    missing_numbers = list(all_numbers - set(numbers))

    return missing_numbers

def iteration(iorRobot, role_list, ITERATION_PATH, department, TASK_TYPE):
    # 选择是否并行、多线程等形式处理数据
    if RUN_TYPE == "parallel":
        iteration_parallel(iorRobot, role_list, ITERATION_PATH, department, TASK_TYPE)
    else: 
        iteration_serial(iorRobot, role_list, ITERATION_PATH, department, TASK_TYPE)
          
def iteration_parallel(iorRobot, role_list, ITERATION_PATH, department, TASK_TYPE):
        # 创建一个进程池
    with Pool(processes=PROSS_NUM) as pool:  # 创建一个包含PROSS_NUM个进程的进程池
        results = [pool.apply_async(process_role, (iorRobot, role, ITERATION_PATH, department, TASK_TYPE)) for role in role_list]  
        # # 异步地将每个role_list中的角色分配给进程池处理，process_role函数会接收相关参数并在异步执行后返回结果
        for r in tqdm(results, desc="Processing samples", unit="sample"):  
            r.wait()  # 显示进度条，等待所有异步任务完成
        pool.close()  # 关闭进程池，不再接受新的任务
        pool.join()  # 等待所有进程完成工作

def iteration_serial(iorRobot, role_list, ITERATION_PATH, department, TASK_TYPE):
    for role in role_list:
        process_role(iorRobot, role, ITERATION_PATH, department, TASK_TYPE)
        
def main(TASK_TYPE):
    iorRobot = ChatBot(MODEL_NAME)
    # 整理路径
    porject_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 获取当前文件目录
    PATH = f"{porject_dir}/dataset" #设置所有这个任务要保存的根路径
    # 输入文件的地址
    INPUT_PATH = f"{PATH}/inputs/final_static.csv"
    # 输出文件的地址
    ITERATION_PATH = f"{PATH}/outputs/IOR-Static/{MODEL_NAME}/{TASK_TYPE}"
    os.makedirs(ITERATION_PATH, exist_ok=True)
    # 开始读取输入文件
    df = read_csv(INPUT_PATH)
    DATASET_NUM = len(df)

    '''为了统计方便，有时需要筛选掉科室数量小的数据,这一步可选择不做'''
    # 统计 'label' 列中每个项的数量
    label_counts = df['lable'].value_counts()
    # 保留个数大于或等于 3 的标签
    labels_to_keep = label_counts[label_counts >= 3].index
    # 筛选 DataFrame，删除数量小于 3 的数据
    filtered_df = df[df['lable'].isin(labels_to_keep)]
    # 提取数据中的科室列表
    department = filtered_df["lable"].unique() 
    # department = df["lable"].unique()

    # 添加id列
    df = df.reset_index().rename(columns={'index': 'id'})
    '''
    因为访问api时会因网络不稳定等原因丢失一些文件，或者运行中断，因此下次再运行整个脚本/检查数据处理量不够时会检查一遍结果目录，若有文件编码丢失会重新跑着一条数据，因此df中的index不建议随便删掉
    '''
    missing_numbers = count_files(ITERATION_PATH, DATASET_NUM)
    df = df.iloc[missing_numbers]
    
    # 把df整理成json格式方便后面调用
    role_list = df_to_dict(df)
    # 开始处理数据
    iteration(iorRobot, role_list, ITERATION_PATH, department, TASK_TYPE)

if __name__ == '__main__':
    for TASK_TYPE in PROMPT_STRATEGIES:
        main(TASK_TYPE)
    
        