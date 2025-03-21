#%%
import os
from openai import OpenAI
from openai import RateLimitError
import time
import random
from dotenv import load_dotenv

load_dotenv(override=True)  # 强制覆盖现有环境变量

class ChatBot:
    def __init__(self,model_name = 'huatuo2') -> None:
        self.key_ind = 0
        self.max_wrong_time = 5
        self.model_name = model_name
        if "huatuo2" in model_name:
            self.url = os.getenv("HUATUO_2_URL")
            self.keys = os.getenv("HUATUO_2_KEY")
        elif "gpt-4" in model_name:
            self.url= os.getenv("GPT_4_URL")
            self.keys = os.getenv("GPT_4_KEY") 
        elif "o1" in model_name:
            self.url= os.getenv("O1_URL")
            self.keys = os.getenv("O1_KEY") 
        elif "triage" in model_name:
            # triage-mix-exp
            self.url= os.getenv("HUATUO_API_URL")
            self.keys = os.getenv("HUATUO_API_KEY") 
        elif "deepseek" in model_name:
            self.url= os.getenv("DEEPSEEK_API")
            self.keys = os.getenv("DEEPSEEK_KEY")
        elif "moonshot" in model_name:
            self.url= os.getenv("MOONSHOT_URL")
            self.keys = os.getenv("MOONSHOT_KEY") 
        else:
            self.url = os.getenv("OR_URL")
            self.keys = os.getenv("OR_KEY")
        assert len(self.keys) > 0, 'have no key'
        self.wrong_time = [0]*len(self.keys)
        print(f'keys: {self.keys}')
        print(f'use model of {self.model_name}')

    def call(self, *args):

        if len(args) == 1:
            message_history = args[0]
            # 单参数逻辑
            message = self.dynamic_call(message_history)
        elif len(args) == 2:
            instruction, prompt = args
            # 双参数逻辑
            message = self.static_call(instruction, prompt)
        else:
            raise TypeError(f"call() 只接受1个或2个参数，当前传入{len(args)}个参数")
        return message

    def static_call(self, instruction, prompt):
        # breakpoint()
        client = OpenAI(
            base_url=self.url,
            api_key=self.keys
        )
        while True:
            try:
                # 初始化OpenAI客户端
                completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt}
                ]
                )
                response = completion.choices[0].message.content
                # 如果返回的消息中包含错误
                if 'error' in response:
                    # 增加当前键对应的错误次数
                    self.wrong_time[self.key_ind] += 1
                    # 如果错误次数超过过最大允许的错误次数
                    if self.wrong_time[self.key_ind] > self.max_wrong_time:
                        # 打印错误响应
                        print(response)
                        # 打印错误的键
                        print(f'Wrong key: {self.keys[self.key_ind]}')
                        # 断言失败，并输出错误信息
                        assert False, str(response)
                # 返回消息内容
                return response
            except RateLimitError as e:
                print(f'Rate limit error: {e}. Retrying...')
                time.sleep(5)  # 等待5秒后重试

    def dynamic_call(self, message_history):
        # breakpoint()
        client = OpenAI(
            base_url=self.url,
            api_key=self.keys
        )
    
        time.sleep(random.randint(1, 2))
        while True:
            # breakpoint()
            try:
                completion = client.chat.completions.create(
                model=self.model_name,
                messages=message_history
                )                
                response = completion.choices[0].message.content
                # print(response)
                # 返回消息内容
                return response
            except RateLimitError as e:
                print(f'Rate limit error: {e}. Retrying...')
                time.sleep(5)  # 等待5秒后重试
                
            except ConnectionError as ae:
                print(f'Connection error: {ae}. Retrying...')
                time.sleep(5)  # 等待5秒后重试
            except Exception as ee:
                print(f'Error: {ee}. Retrying...')
                print(message_history)
                break
