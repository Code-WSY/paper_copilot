from openai import OpenAI
import os
from dotenv import load_dotenv
from src.vector_indexer import VectorIndexer
from termcolor import colored
#Markdown优化输出
from rich.markdown import Markdown
from rich import print as rprint
import json
import time
load_dotenv()

class Agent:
    def __init__(self,prompt_path="src/prompt/文献分析助手.md",model="o1-mini"):
        self.client = OpenAI(api_key=os.getenv("API_KEY"),base_url=os.getenv("BASE_URL"))
        self.prompt = self.load_prompt(prompt_path)
        self.model = os.getenv("MODEL")
        self.top_n=int(os.getenv("TOP_N"))
        self.relation_threshold=float(os.getenv("RELATION_THRESHOLD"))
        self.history = []
        self.vector_indexer = VectorIndexer()
        self.vector_indexer.select_tables()
        self.last_response = None

    def load_prompt(self, prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read()
        
    def save_chat_history(self):
        chat_history_dir = "chat_history"
        if not os.path.exists(chat_history_dir):
            os.makedirs(chat_history_dir)
        with open(f"{chat_history_dir}/chat_history_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json", "w", encoding="utf-8") as file:
            #history字符似乎显示乱码需要转码
            json.dump(self.history, file,ensure_ascii=False)
        print(colored(f"已保存聊天记录:{file}", "green"))

    def load_chat_history(self):
        chat_history_dir = "chat_history"
        if not os.path.exists(chat_history_dir):
            print(colored("没有聊天记录", "red"))
            return
        print(colored(f"聊天记录：", "blue"))
        for i, file in enumerate(os.listdir(chat_history_dir), 1):
            print(f"{i}. {file}")
        #让用户选择序号
        choice = int(input("请输入序号: "))
        #加载用户选择的聊天记录
        with open(os.path.join(chat_history_dir, os.listdir(chat_history_dir)[choice-1]), "r", encoding="utf-8") as file:
            self.history = json.load(file)
        print(colored(f"已加载聊天记录", "green"))
        
    def clear_chat_history(self):
        self.history = []
        print(colored(f"已清除当前聊天记录", "green"))

    def decorate_user_input(self, user_input, related_docs):
        # 将相关文档内容添加到用户输入中
        for cos,table_name,doc in related_docs:
            print(colored(f"检索到的文件标题：{table_name} ; 相关度：{cos}", "green"))
            user_input += f"\n\n相关文件标题：{table_name}\n该文件的相关内容：{doc}"
        return user_input

    def get_response_of_vector_database(self, user_input,history=None):
        # 1. 根据用户输入，在向量数据库中检索相关文档
        # 2. 将检索到的文档内容返回给LLM，LLM根据文档内容和用户问题，生成回答
        # 3. 返回回答
        related_docs = self.vector_indexer.search_index(user_input)
        #提取关联度>relation_threshold 的文档
        related_docs = [(cos,table_name,doc) for cos,table_name,doc in related_docs if cos > self.relation_threshold]
        print(colored(f"检索到{len(related_docs)}个相关部分", "green"))
        # 修饰问题
        user_input = self.decorate_user_input(user_input, related_docs)
        if len(self.history)==0:
            #加入提示词 放在user里
            user_input = self.prompt +'\n\n'+ "用户请求："+user_input
            self.history.append({"role": "user", "content": user_input})
        else:
            self.history.append({"role": "user", "content": user_input})
        #print(colored(f"修饰后的用户输入: {user_input}", "blue"))
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history
        )
        self.last_response = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": self.last_response})
        return self.last_response

    def get_response_of_ai(self,user_input):
        self.history.append({"role": "user", "content": user_input})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history
        )
        self.last_response = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": self.last_response})
        return self.last_response

    def chat_with_vector_database(self,user_input):
        self.get_response_of_vector_database(user_input)
        rprint("Assistant: \n", Markdown(self.last_response))
        print(colored("-"*50, "green"))

    def chat_with_ai(self,user_input):
        self.get_response_of_ai(user_input)
        rprint("Assistant: \n", Markdown(self.last_response))
        print(colored("-"*50, "green"))

    def save_last_response(self):
        answer_dir = "answer"
        save_path = f"{answer_dir}/answer_{time.strftime('%Y-%m-%d_%H-%M-%S')}.md"
        if self.last_response is None:
            print(colored("没有上一次的回答", "red"))
            return
        if not os.path.exists(answer_dir):
            os.makedirs(answer_dir)
        with open(save_path, "w", encoding="utf-8") as file:
            file.write(self.last_response)
        print(colored(f"已保存上一次的回答到：{save_path}", "green"))
        print(colored("-"*50, "green")) 

    def delete_table(self):
        self.vector_indexer.delete_table()

    def reselect(self):
        self.vector_indexer.select_tables()

if __name__ == "__main__":
    agent = Agent(prompt_path="src/prompt/文献分析助手.md",model="o1-mini")
    agent.chat_with_vector_database()
