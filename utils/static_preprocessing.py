import pandas as pd
import os
import numpy as np
import re
import ast


def read_xlsx(file_path):
    # 读取.xlsx文件到DataFrame里
    df = pd.read_excel(file_path)
    print(df.columns)
    print(df.head(5))
    # len(df) = 511405
    return df

def read_csv(file_path):
    # # 读取.xlsx文件到DataFrame里
    # 使用pandas的read_excel函数读取Excel文件
    df = pd.read_csv(file_path)
    print(df.columns)
    print(df.head(5))
    # len(df) = 511405
    return df

def read_json(file_path):
    # 使用pandas的read_json函数读取JSON文件
    df = pd.read_json(file_path)
    print(df.columns)
    print(df.head(5))
    return df

def save_csv(df, file_path):
    # 保存DataFrame到CSV文件中
   
    # 使用pandas的to_csv函数将DataFrame保存到CSV文件
    df.to_csv(file_path, index=False)

    # 输出保存成功的信息
    print(f'DataFrame已成功保存到 {file_path}')

def convert_age(age_str):
    if pd.isna(age_str) or not isinstance(age_str, str):
        return np.nan
    if '岁' in age_str:
        try:
            return float(re.sub(r'岁', '', age_str))
        except ValueError:
            return np.nan
    elif '月' in age_str and '天' in age_str:
        month_day = re.match(r'(\d+)月(\d+)天', age_str)
        if month_day:
            months = int(month_day.group(1))
            days = int(month_day.group(2))
            return round(months + days / 30, 2)  # 每月按30天计算
    elif '月' in age_str:
        try:
            months = int(re.sub(r'月', '', age_str))
            return round(months / 12, 2)  # 每年按12个月计算
        except ValueError:
            return np.nan
    elif '天' in age_str:
        try:
            days = int(re.sub(r'天', '', age_str))
            return round(days / 365, 2)  # 每年按365天计算
        except ValueError:
            return np.nan
    return np.nan

# 定义函数来执行这两步操作
def process_dialog_content(content):
    # 将字符串转换为列表
    list_data = ast.literal_eval(content)
    # 删除列表中的空值
    cleaned_list = [item for item in list_data if len(item) != 0]
    return cleaned_list

def clear_data(df):
    # 选择构建导诊系统需要的list
    # base_list = ['挂号科室', '就诊科室', '年龄', '性别', '对话内容', '对话1', '对话2', '对话3',
    #    '对话4', '对话5', '对话6', '对话7', '对话8', '对话9', '对话10', '对话11', '对话12',
    #    '对话13', '对话14', '对话15']
    base_list = ['年龄', '性别', '对话内容', 'lable']
    # 使用ast.literal_eval将字符串转换为列表
    # list_data = ast.literal_eval(df['对话内容'][0])
    # # [item for item in list_data if len(item)!=0]
    
    df['对话内容'] = df['对话内容'].apply(process_dialog_content)
    
    selected_columns = [element for element in df.columns if element in base_list]
    new_df = df[selected_columns]
    # breakpoint()
    # 对'对话内容'列应用上述函数并将结果存储在新的'对话轮数'列中
    new_df['对话轮数'] = new_df['对话内容'].apply(process_and_count)
    # breakpoint()
    return new_df

def extract_data(df):
    breakpoint()
    # 确定抽样总数
    total_samples = 100
    # 按'科室'列进行分组
    grouped = df.groupby('lable')
    # 计算每组应该抽取的样本数
    group_sizes = grouped.size()
    group_ratios = group_sizes / group_sizes.sum()
    samples_per_group = (group_ratios * total_samples).round().astype(int)
    # breakpoint()
    # 从每组中按比例抽取样本
    sampled_df = grouped.apply(lambda x: x.sample(n=samples_per_group[x.name])).reset_index(drop=True)
    # breakpoint()
    # 按'科室'列进行分组，并过滤掉每组数量少于5的组
    # filtered_df = sampled_df.groupby('科室名称').filter(lambda x: len(x) >= 5)
    return sampled_df

def filter_data(df):
    # # bucm_sz
    # list_department = ['临床心理门诊', '儿内科', '儿童保健科', '内分泌科门诊', '口腔科门诊',
    #    '妇科门诊', '心病科门诊', '急诊科', '感染性疾病科门诊', '推拿科门诊', '普通外科',
    #    '正骨门诊', '治未病科门诊', '泌尿外科', '生殖健康科门诊', '生殖男科门诊', '皮肤科门诊', '眼科门诊',
    #    '神经外科', '耳鼻喉科门诊', '肛肠科门诊', '肝病科门诊', '肺病科门诊', '肾病科门诊', '肿瘤科门诊',
    #    '脑病科门诊', '脾胃病科门诊', '针灸科门诊', '门诊内科', '风湿病科门诊', '骨伤科门诊']
    # zxyy
    # list_department = ['临床心理门诊', '产科门诊', '保健科门诊', '儿内科门诊', '儿童保健科',
    #    '关节组门诊', '内分泌专科门诊', '创伤骨科门诊', '口腔科门诊', '呼吸与危重症医学科门诊',
    #    '妇科门诊', '康复科门诊', '心胸外科门诊', '心血管内科门诊', '急诊耳鼻喉', '感染科门诊',
    #    '整形外科门诊', '普外科门诊', '泌尿外科门诊', '消化内科门诊',
    #    '烧伤外科门诊', '生殖医学科门诊', '甲乳血管外科门诊', '疼痛科门诊', '皮肤科', '眼科门诊', '神经内科门诊',
    #    '神经外科门诊', '耳鼻咽喉门诊', '肛肠外科门诊', '肾内科门诊',
    #    '肿瘤内科门诊', '脊柱外科门诊', '血液内科门诊',
    #    '风湿免疫科门诊', '骨关节科门诊']
    # breakpoint()
    # static
    list_department = ['肛肠外科', '甲状腺乳腺外科', '妇科', '脊柱外科', '耳鼻喉科', '消化内科', '内分泌代谢科', '门诊产科',
       '泌尿外科', '神经内科', '心血管内科', '创伤骨科与骨关节科', '口腔科', '妇科内分泌科',
       '消化内科、神经内科', '心胸外科', '眼科', '门诊妇科', '风湿免疫科门诊', '呼吸内科', '皮肤科',
       '运动医学与骨关节科', '神经外科', '肾内科', '儿科', '肝胆胰脾疝外科', '烧伤整形科',
       '普外科', '临床心理门诊', '血液内科', '感染性疾病科', '产科', '康复医学科', '生殖医学科', '精神心理科',
       '新生儿科', '肝胆外科', '肝胆胰疝外科', '普通外科', '皮肤性病科', '血管外科', '疼痛科', '肝病门诊',
       '风湿免疫科']
    
    # 只保留科室名称在 listA 中的数据
    new_df = df[df['lable'].isin(list_department)]
    # breakpoint()
    return new_df

def df_to_dict(df):
    # 将每一行转换成字典，并存储在一个列表中
    dict_list = df.to_dict('records')
    return dict_list

def get_role_prompt(role_dict):
    age = role_dict['年龄']
    sex = role_dict['性别']
    chief_complaint = role_dict['主诉']
    history_of_present_illness = role_dict['现病史']
    history_of_past_illness = role_dict['既往史']
    personal_history = role_dict['个人史']
    family_history = role_dict['家族史']
    if age >= 18:
        role = f"一个{age}岁的{sex}病人"
        patient_info = f'你生病的情况'
        scenario = f"你来医院看病，不知道挂号哪个科室。"
    else:
        role = f"一个{age}岁{sex}孩的家长"
        patient_info = f"孩子生病的情况"
        scenario = f"你带孩子来医院看病，不知道挂号哪个科室。"
    role_prompt = f'''
    对话场景：{scenario}
    你的角色：{role}
    角色相关信息：{patient_info}: 1. 主要症状：{chief_complaint}，2. 现病史：{history_of_present_illness}，3. 既往史：{history_of_past_illness}，4. 个人史：{personal_history}，5. 家族史：{family_history}。
    你的任务：你来医院看病，不知道挂号哪个科室，你要根据给定的角色信息回答医生的提问。
    任务注意事项：1. 你只回答医生提问的信息，不可杜撰。2. 每个回答都只提供1-2个信息，不可一次把所有症状描述出来。3. 你不是医疗专业人士，不懂也不会说专业词汇。
    '''
    role_dict['patient_prompt'] = role_prompt
    return role_dict

# 定义函数来执行这两步操作并返回列表长度
def process_and_count(cleaned_list):
    
    return len(cleaned_list)

def determine_label(row):
    if pd.notna(row['标注科室']):
        return row['标注科室']
    elif row['预约科室'] == row['挂号科室'] == row['就诊科室']:
        return row['挂号科室']
    else:
        return None
    
def role_process(df):
    dict_list = df_to_dict(df)
    new_list = []
    for role_dict in dict_list:
        new_list.append(get_role_prompt(role_dict))
    final_df = pd.DataFrame(new_list)
    return final_df
    
def main():
     
    # 获取当前工作目录
    current_path = os.getcwd()

    # # 打印当前工作目录
    print(f'当前工作目录是: {current_path}')
    
    # INSTITUTE_NAME = "zxyy"
    TYPE = "Static" # dynamic or static
    # PATH = f"{current_path}/dataset/{INSTITUTE_NAME}"
    # # INPUT_PATH = f"{PATH}/zxyy.xlsx"
    # INPUT_PATH = f"{PATH}/{INSTITUTE_NAME}_{TYPE}_5000.xlsx"
    INSTITUTE_NAME = "zxyy"
    # TYPE = "dynamic" # dynamic or static
    PATH = f"{current_path}/dataset/{TYPE} dataset"
    # "annotation_bucm"
    INPUT_PATH = f"{PATH}/static_1000.xlsx"
    
    INPUT_PATH = f"{PATH}/final_static_616.csv"
    OUTPUT_PATH = f"{PATH}/human_eval.csv"
    # OUTPUT_PATH_with_prompt = f"{PATH}/{INSTITUTE_NAME}_{TYPE}_prompt.csv"
    # org_df = read_xlsx(INPUT_PATH)
    
    org_df = read_csv(INPUT_PATH)
    breakpoint()
    # 更改列名 'old_name' 为 'new_name'
    # org_df.rename(columns={'Unnamed: 6': '标注科室'}, inplace=True)
    # org_df['lable'] = org_df.apply(determine_label, axis=1)
    # # 删除 'label' 列为 None 的行
    # org_df = org_df.dropna(subset=['lable'])

    # breakpoint()
    # org_df['年龄'] = org_df['年龄'].replace({'Y': '岁', 'M': '个月', 'D': '天'}, regex=True)
    # df = filter_data(org_df)
    # # org_df = read_csv(INPUT_PATH)
    # df = clear_data(df)
    
    df = extract_data(org_df)
    save_csv(df, OUTPUT_PATH)
    # role_df = role_process(df)

if __name__ == '__main__':
    main()    