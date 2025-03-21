import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, BertForSequenceClassification, BertTokenizer
import evaluate
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
import torch
# from sklearn.model_selection import train_test_split


# # 加载数据集
# data_path = 'final_static_617.csv'
# dataset = pd.read_csv(data_path)

# # 解析并合并对话内容
# def merge_dialogue(dialogue):
#     try:
#         dialogue_list = json.loads(dialogue.replace("'", '"'))
#         merged_dialogue = " ".join([d['病人'].strip() + " " + d['医生'].strip() for d in dialogue_list])
#         # breakpoint()
#         return merged_dialogue
#     except:
#         return ""
    
# def remove_last_doctor_reply(dialogue: str) -> str:
#     # 将对话按换行符分割为列表
#     lines = dialogue.strip().split('\n')
#     # 逆序遍历对话行，找到最后一个医生的回复
#     for i in range(len(lines)-1, -1, -1):
#         if lines[i].startswith('医生:'):
#             # 去掉最后一个医生的回复并返回
#             return '\n'.join(lines[:i])
#     # 如果没有找到医生的回复，返回原对话
#     return dialogue

# def process_dialogue(input_str):
#     # breakpoint()
#     # 将字符串中的单引号替换为双引号
#     # input_str = input_str.replace("'", '"').replace('''""''', '"').replace('“', '"')
#     input_str = input_str.replace("'", '"')
    
#     # breakpoint()
#     # print(input_str)
    
#     dialogues = json.loads(input_str)
#     # 格式化输出
#     formatted_dialogue = ""
#     for dialogue in dialogues:
#         for role, sentence in dialogue.items():
#             formatted_dialogue += f"{role}: {sentence}\n"
        
#     # print(formatted_dialogue)
#     return formatted_dialogue

# # 应用解析函数
# # dataset['合并对话'] = dataset['对话内容'].apply(merge_dialogue)
# dataset['合并对话'] = dataset['对话内容'].apply(process_dialogue)
# dataset["input"] = dataset['合并对话'].apply(remove_last_doctor_reply)
# # breakpoint()
# # dialouge_history = remove_last_doctor_reply(process_dialogue(role["对话内容"])) #提取历史对话记录
# # 随机抽取20%的数据作为测试集，其余80%作为训练集
# train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)

# # 保存训练集和测试集到CSV文件
# train_df.to_csv('train.csv', index=False)
# test_df.to_csv('test.csv', index=False)

# 加载数据集
data_path = 'train.csv'
dataset = pd.read_csv(data_path)

# 初始化LabelEncoder
label_encoder = LabelEncoder()

# 转换标签列
dataset['labels'] = label_encoder.fit_transform(dataset['lable'])

# 将DataFrame转换为Hugging Face数据集
hf_dataset = Dataset.from_pandas(dataset[['input', 'labels']])

model_name = "bert-base-chinese"
# 使用预训练的分词器
tokenizer = BertTokenizer.from_pretrained(model_name)

# 分词函数
def tokenize_function(examples):
    return tokenizer(
        examples['input'],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# 应用分词
tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)

# 按8:2比例切分数据集
train_test_split = tokenized_datasets.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']


# 加载序列分类模型
num_labels = dataset['labels'].nunique()
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练参数
training_args = TrainingArguments(
    per_device_train_batch_size=64,  # 训练时的batch_size
    per_device_eval_batch_size=64,  # 验证时的batch_size
    logging_steps=100,                # log 打印的频率
    evaluation_strategy="epoch",     # 评估策略
    num_train_epochs = 50,            # 训练epoch数
    output_dir="test_trainer",
    save_strategy="epoch",           # 保存策略
    save_total_limit=1,  
    load_best_model_at_end=True,
    push_to_hub=True,
)

# 评估指标
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 初始化训练器并进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    
)

trainer.train()
trainer.push_to_hub()
# 模型评估
results = trainer.evaluate(eval_dataset)
print(f"Evaluation results: {results}")
# 模型测试
# predictions = trainer.predict(eval_dataset)
# print(f"Prediction results: {predictions.predictions}")

print("All done!")

# 模型预测
print("Predicting on test dataset...")
predictions = trainer.predict(eval_dataset)

# 获取预测的 logits 和真实标签
logits = predictions.predictions
labels = predictions.label_ids

# 计算每个样本的预测标签
predicted_labels = np.argmax(logits, axis=-1)

# 计算预测正确的数量
correct_predictions = (predicted_labels == labels).sum()
total_predictions = len(labels)

# 输出预测结果和正确数量
print(f"Total predictions: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {correct_predictions / total_predictions:.4f}")