import pandas as pd
import os
import re
import argparse
import json
from utils.simulator import ChatBot
from multiprocessing import Pool
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Data Analysis")
parser.add_argument('--max_turn', type=int, default=7, help='The max turn of the conversation' )
parser.add_argument('--model_name', type=str, default="gpt-4o", help='The name of the model')
parser.add_argument('--run_type', type=str, default="normal", help='parallel or normal')
parser.add_argument('--process_num', type=int, default=20, help='The max turn of the conversation' )

args = parser.parse_args()
MAX_TURN = args.max_turn
MODEL_NAME = args.model_name
queryRobot = ChatBot(MODEL_NAME) 
patientRobot = ChatBot("gpt-4o")
iorRobot = ChatBot(MODEL_NAME) #探针
RUN_TYPE = args.run_type
PROCESS_NUM = args.process_num

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
    # 找出0到DATASET_NUM之间的所有数字
    all_numbers = set(range(int(DATASET_NUM)))
    # 找出numbers列表中没有的数字
    missing_numbers = list(all_numbers - set(numbers))
    return missing_numbers

def get_role_prompt(role_dict): 
    age = role_dict['年龄']
    sex = role_dict['性别']
    chief_complaint = role_dict['主诉']
    history_of_present_illness = role_dict['现病史']
    history_of_past_illness = role_dict['既往史']
    family_history = role_dict['家族史']
    if age >= 12.0:
        role = f"一个{age}岁的{sex}病人"
        patient_info = f'你生病的情况是'
        scenario = f"你来医院看病，不知道挂号哪个科室。"
    else:
        role = f"一个{age}岁{sex}孩的家长"
        patient_info = f"孩子生病的情况是"
        scenario = f"你带孩子来医院看病，不知道挂号哪个科室。"
    role_prompt = f'''
    你是{role}, {scenario}你的任务是，与医生交流，根据他的提问来回答。{patient_info}: 1. 主要症状：{chief_complaint}2. 现病史：{history_of_present_illness} 3. 既往史：{history_of_past_illness}，4. 家族史：{family_history}。
    注意：
    1. 你必须如实回答，不可杜撰。
    2. 每个回答只提供1-3个主要信息，不可一次把所有症状描述出来。
    3. 你不是医疗专业人士，不懂也不会说专业词汇。
    '''
    role_dict['patient_prompt'] = role_prompt
    return role_dict

def get_ior_consider_prompt(role_dict):
    age_num = role_dict['年龄']
    age = role_dict['年龄']
    sex = role_dict['性别']
    if age_num >= 12.0:
        role = f"一个{age}的{sex}病人"
    else:
        role = f"一个{age}{sex}孩的家长，带孩子来看病"
    ior_consider_template = f'''
    你是一位导诊过程中的提问医生，为了引导病人尽量多地给出病症相关信息，你要一直向病人提问与科室推荐相关的关键信息。
    注意：
    1. 你的输出必须是问题。
    2. 每次只提出1-2个问题，不要提问多个问题。
    3. 病人不懂专业词汇，所以你的问题必须易于理解，符合口语习惯。
    4. 不要一个症状一直问，当你判断当前症状已经获取足够信息，你要换其他的关键信息提问。
    4. 你面对的是{role}。
    '''
    return ior_consider_template

def force_ior(department, role_dict):
    age_num = role_dict['年龄']
    age = role_dict['年龄']
    sex = role_dict['性别']
    if age_num >= 12.0:
        role = f"一个{age}的{sex}病人"  
    else:
        role = f"一个{age}{sex}孩子的家长，带孩子来看病"
    force_ior_instrution = f'''
    你是医院分诊台的导诊医生，负责给病人推荐合适的科室。你负责的科室是：
    {str(department)}。此时你正在面对{role}。
    根据所有你和病人的对话信息，你要仔细思考他可能要去看病的科室是什么，从科室表中为他推荐一个科室。科室名称必须完全和上面一致，必须一字不差。
    注意：
    1.你绝不能补全后面好几轮的对话！！
    2.你只能从分诊科室表中推荐一个最合适的科室，不要推荐多个科室。
    3.绝对不能提问，必须推荐一个科室！！
    '''
    return force_ior_instrution

def detect_results(item_list, target_string):
    try:
        found_items = [item for item in item_list if item in target_string]
        if len(found_items) == 0:
            return ""
        else:
            return found_items[0]
    except:
        return ""

def process_role_prompt(ziji = "", role="", messages="",content=""):
    if "patient" in ziji:
        if "patient" in role:
            role_content = {"role" : "assistant", "content": content}
        elif "doctor" in role:
            role_content = {"role" : "user", "content": content}
        else:
            role_content = {"role" : "user", "content": "不好意思，我没听清，您能再说一下吗？"}
    elif "doctor" in ziji:
        if "patient" in role:
            role_content = {"role" : "user", "content": content}
        elif "doctor" in role:
            role_content = {"role" : "assistant", "content": content}
        else:
            role_content = {"role" : "user", "content": "不好意思，我没听清，您能再说一下吗？"}
    else:
        role_content = {"role" : "user", "content": "不好意思，我没听清，您能再说一下吗？"}
    messages.append(role_content)
    return messages

def interact(patient_instruction, ior_consider_template, force_ior_instruction, max_turn, department, label):
    doc_role_prompt = {"role": "assistant", "content": "您好，请问有什么可以帮您？"}
    messages_doc_query = [{"role": "system", "content": ior_consider_template}, doc_role_prompt]
    messages_patinet_query = [{"role": "system", "content": patient_instruction}, {"role": "user", "content": "您好，请问有什么可以帮您？"}]
    messages_doc_probe = [{"role": "system", "content": force_ior_instruction}, doc_role_prompt]
    DIALOG = "您好，请问有什么可以帮您？"
    count = 0
    probe_dict = {}
    while count < max_turn:
        # 开启对话
        response = patientRobot.call(messages_patinet_query) # 获取病人第一轮信息
        
        # --整理prompt--
        messages_doc_query = process_role_prompt("doctor","patient", messages_doc_query, response) #反问医生
        messages_patinet_query = process_role_prompt("patient", "doctor",messages_patinet_query, response) #病人
        messages_doc_probe = process_role_prompt("doctor", "patient", messages_doc_probe, response) # 探针医生
        # --记录对话历史--
        DIALOG += f"病人：{response}\n "

        # 医生进行提问
        response = queryRobot.call(messages_doc_query)
        # --整理prompt--
        messages_doc_query = process_role_prompt("doctor", "doctor", messages_doc_query, response)
        messages_patinet_query = process_role_prompt("patient", "patient",messages_patinet_query, response)
        # 
        # --记录对话历史--
        DIALOG += f"医生：{response}\n "
        
        # 指针收集导诊结果
        probe_ior = iorRobot.call(messages_doc_probe)
        messages_doc_probe = process_role_prompt("doctor", "doctor", messages_doc_probe, response)
        tri_res = detect_results(department, probe_ior)
        probe_dict[str(count)] = tri_res
        k_n = f"sucess_{count}"
        probe_dict[k_n] = 1 if label in tri_res else 0
        count+=1
    return DIALOG, probe_dict

def process_role(role, ITERATION_PATH, department, max_turn, INSTITUTE):
    patient_instruction = get_role_prompt(role)['patient_prompt']
    ior_consider_template = get_ior_consider_prompt(role)
    force_ior_instruction = force_ior(department, role)
    curr_dep = role[INSTITUTE]
    try:
        dialogue, probe_dict = interact(patient_instruction, ior_consider_template, force_ior_instruction, max_turn, department, curr_dep)
        # 在这里添加处理角色的逻辑
        # 如果发生错误，引发APIStatusError
        # raise APIStatusError(response="Some response", body="Some body")
        pass  # 替换为实际处理逻辑
    except Exception as e:
        logging.error(f"Error processing role {role}: {e}")
        raise
    role['dialogue'] = dialogue
    role.update(probe_dict)
    # 把当前一轮的结果先存到json临时文件中
    role_path = f"{ITERATION_PATH}/role_{role['id']}.json"
    print(role_path)
    save_json(role, role_path)

def iteration(role_list, ITERATION_PATH, department, max_turn, INSTITUTE):
    if RUN_TYPE == "parallel":
        iteration_parallel(role_list, ITERATION_PATH, department, max_turn, INSTITUTE)
    elif RUN_TYPE == "multi_process":
        iteration_multi_process(role_list, ITERATION_PATH, department, max_turn, INSTITUTE)
    else:
        iteration_serial(role_list, ITERATION_PATH, department, max_turn, INSTITUTE)

# 多线程 进行改写
def iteration_multi_process(role_list, ITERATION_PATH, department, max_turn, INSTITUTE):
    # max_workers = os.cpu_count()  
    with ThreadPoolExecutor(max_workers=PROCESS_NUM) as executor:
        futures = [executor.submit(process_role, role, ITERATION_PATH, department, max_turn, INSTITUTE) for role in role_list]
        # 使用 tqdm 显示进度
        for future in tqdm(futures, desc="Processing samples", unit="sample"):
            try:
                future.result()  # 等待每个线程完成并获取结果
            except Exception as e:
                print(f"发生错误: {e}")

def iteration_parallel(role_list, ITERATION_PATH, department, max_turn, INSTITUTE):
        # 创建一个进程池
    with Pool(processes=PROCESS_NUM) as pool:  # 创建一个包含n个进程的进程池
        results = [pool.apply_async(process_role, (role, ITERATION_PATH, department, max_turn, INSTITUTE)) for role in role_list]  
        # 异步地将每个role_list中的角色分配给进程池处理，process_role函数会接收相关参数并在异步执行后返回结果
        for r in tqdm(results, desc="Processing samples", unit="sample"):  
            r.wait()  # 显示进度条，等待所有异步任务完成
        pool.close()  # 关闭进程池，不再接受新的任务
        pool.join()  # 等待所有进程完成工作

def iteration_serial(role_list, ITERATION_PATH, department, max_turn, INSTITUTE):
    result_list = []
    for role in role_list:
        process_role(role, ITERATION_PATH, department, max_turn, INSTITUTE)

        result_list.append(role)
    return result_list

def main(INSTITUTE):
    # 整理路径
    porject_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 获取当前文件目录
    PATH = f"{porject_dir}/dataset" #设置所有这个任务要保存的根路径
    # 输入文件的地址
    INPUT_PATH = f"{PATH}/inputs/dynamic_final.csv"
    # 输出文件的地址
    ITERATION_PATH = f"{PATH}/outputs/IOR-Dynamic/{INSTITUTE}/{MODEL_NAME}"
    os.makedirs(ITERATION_PATH, exist_ok=True)
    # 开始读取输入文件
    df = read_csv(INPUT_PATH)
    DATASET_NUM = len(df)
    # 提取数据中的科室列表d
    department = df[INSTITUTE].unique()
    # 添加id列
    df = df.reset_index().rename(columns={'index': 'id'})        
    missing_numbers = count_files(ITERATION_PATH, DATASET_NUM)
    df = df.iloc[missing_numbers]
    # 把df整理成json格式方便后面调用
    role_list = df_to_dict(df)
    # 开始处理数据
    iteration(role_list, ITERATION_PATH, department, MAX_TURN, INSTITUTE)

if __name__ == '__main__':
    institutes = ["hospital-1","hospital-2"]
    for institute in institutes:
        main(institute)
    # main()
    