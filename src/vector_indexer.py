from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
from src.utils import extract_text_from_pdf,extract_text_from_txt,extract_text_from_md,extract_text_from_docx
#优化输出colored
from termcolor import colored
#rprint
from rich import print as rich_print
#sqlite3
import sqlite3
#tqdm
from tqdm import tqdm
#pickle
import pickle
#log
import logging
import time
#日志信息
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s') #INFO级别，输出所有信息
load_dotenv()

class VectorIndexer:
    def __init__(self,database_path=None):
        self.database_path = database_path or os.getenv("DATABASE_PATH")
        self.client = OpenAI(api_key=os.getenv("API_KEY"),base_url=os.getenv("BASE_URL"))
        self.batch_size = int(os.getenv("BATCH_SIZE"))
        self.top_n = int(os.getenv("TOP_N"))
        self.tables = []
        self.parallel_num = int(os.getenv("PARALLEL_NUM"))
        #如果数据库路径不存在，则创建
        if not os.path.exists(self.database_path):
            print(colored("数据库路径不存在，是否创建？(y/n)", "red"))
            choice = input().strip()
            if choice.lower() == 'y':
                os.makedirs(os.path.dirname(self.database_path),exist_ok=True)
                #logging.info(colored("数据库路径创建成功", "green"))
                print(colored("正在创建数据库", "green"))
                self.load_index()
            else:
                print(colored("取消创建数据库，程序退出", "red"))
                exit(0)

    def encode(self, text,sleep=0):
        response = self.client.embeddings.create(input=text, model="text-embedding-3-large")
        time.sleep(sleep)
        return np.array(response.data[0].embedding)
    
    def select_tables(self):
        # 先列出所有除了sqlite_sequence表的表名
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name!='sqlite_sequence'")
        tables = [row[0] for row in cursor.fetchall()]
        if not tables:
            print(colored("没有找到任何文献。", "yellow"))
            self.tables = []
            return
        # 显示表名列表
        print("-"*50)
        print(colored("数据库中已有的文献:", "blue"))
        print("-"*50)
        for idx, table in enumerate(tables, start=1):
            print(colored(f"{idx}. {table}", "blue"))
        print("-"*50)
        # 让用户选择序号，分号;隔开，或者:表示连续 如1:5;7;9表示1,2,3,4,5,7,9
        user_input = input(colored("请选择文献的编号（例如1:5;7;9;all）: ", "cyan"))
        selected_indices = set()
        #ALL表示所有表
        if user_input.strip().lower() == 'all':
            self.tables = tables
        #如果直接回车
        elif user_input.strip() == '':
            self.tables = []
            print("-"*50)
            print(colored("未选择任何文献", "yellow"))
            print("-"*50)
            return
        else:
            for part in user_input.split(';'):
                if ':' in part:
                    start, end = part.split(':')
                    selected_indices.update(range(int(start), int(end)+1))
                else:
                    selected_indices.add(int(part))
            # 过滤无效的索引
            selected_indices = {i for i in selected_indices if 1 <= i <= len(tables)}
            selected_tables = [tables[i-1] for i in sorted(selected_indices)]
            self.tables = selected_tables

            if len(self.tables) == 0:
                print("-"*50)
                print(colored("未选择任何文献", "yellow"))
                print("-"*50)
                return
        
        print("-"*50)
        print(colored("已选择文献:", "blue"))
        for idx,table in enumerate(self.tables, start=1):
            print(colored(f"{idx}. {table}", "blue"))
        print("-"*50)

        conn.close()      

    def cal_cos(self, input_vec, embedding):
        """计算余弦相似度"""
        try:
            #如果输入是字符串，则将其转换为numpy数组
            if isinstance(input_vec, str):
                input_vec = np.fromstring(input_vec.strip('[]'), sep=' ')
            else:
                input_vec = input_vec.astype(float)
            #如果输入是字符串，则将其转换为numpy数组
            if isinstance(embedding, str):
                embedding = np.fromstring(embedding.strip('[]'), sep=' ')
            else:
                embedding = embedding.astype(float)
            
            numerator = np.dot(input_vec, embedding)
            denominator = np.linalg.norm(input_vec) * np.linalg.norm(embedding)
            if denominator == 0:
                return 0.0
            return numerator / denominator
        except Exception as e:
            print(colored(f"转换向量时出错: {e}", "red"))
            return 0.0

    def dir_to_text_vec(self, dir_path):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        text_vec_list = []
        with ThreadPoolExecutor(max_workers=self.parallel_num) as executor:
            # 收集所有文件路径
            futures = {executor.submit(self.file_to_text_vec, os.path.join(root, file)): file 
                       for root, dirs, files in os.walk(dir_path) for file in files}
            # 遍历所有文件
            for future in tqdm(as_completed(futures), desc=colored("总进度", "green")):
                # 获取文件名
                file = futures[future]
                try:
                    # 获取文件向量
                    text_vec = future.result()
                    # 如果文件向量不为空
                    if text_vec is not None:
                        # 只取文件名并替换特殊字符
                        file_name = os.path.basename(file).replace('.', '_').replace(':', '_').replace(' ', '_')
                        text_vec_list.append((file_name, text_vec))
                except Exception as e:
                    print(colored(f"处理文件{file}时出错: {e}", "red"))
                    continue
        # 如果文件向量列表为空
        if len(text_vec_list) == 0:
            print(colored("没有有效的文件", "red"))
            return None

        return text_vec_list
    
    def file_to_text_vec(self,file_path):
        #检查是否存在
        table_name=os.path.basename(file_path).replace('.','_').replace(':','_').replace(' ','_')
        if self.check_table_exist(table_name):
            print(colored(f"文件 {file_path} 已存在", "dark_grey"))
            return None
        # 将文件内容转换为向量，只读取pdf，txt，md，docx文件
        enable_file_types = ['pdf', 'txt', 'md', 'docx']
        if not any(file_path.endswith(ext) for ext in enable_file_types):
            print(colored(f"文件 {file_path} 不是有效的文件类型", "dark_grey")) 
            return None
        # 将文件内容转换为向量
        #print(colored(f"正在处理文件: {file_path}", "green"))
        if file_path.endswith('.pdf'):
            text=extract_text_from_pdf(file_path)
        elif file_path.endswith('.txt'):
            text=extract_text_from_txt(file_path)
        elif file_path.endswith('.md'):
            text=extract_text_from_md(file_path)
        elif file_path.endswith('.docx'):
            text=extract_text_from_docx(file_path)

        text_vec=self.text_to_vec(text,file_path)
        return text_vec
    
    def text_to_vec(self,text,file_path):
        text_vec = []
        print(colored(f"正在为文件：{os.path.basename(file_path)}生成向量", "green"))
        for i in tqdm(range(0, len(text), self.batch_size), desc=colored(f"进度", "green")):
            batch = text[i:i+self.batch_size]
            embedding = self.encode(batch,sleep=1)
            text_vec.append({"content": batch, "embedding": embedding})
        return text_vec
    
    def create_index(self, documents_path):
        """
        创建向量索引
        输出：
           [(文件路径1,[(content,embedding),(content,embedding),...]),
            (文件路径2,[(content,embedding),(content,embedding),...]),
            ...]
        """
        if os.path.isdir(documents_path):
            print(colored("正在处理文件夹: " + documents_path, "green"))
            text_vec=self.dir_to_text_vec(documents_path)
        else:
            print(colored("正在处理文件: " + documents_path, "green"))
            text_vec=self.file_to_text_vec(documents_path)
            if text_vec is None:
                return None
            text_vec=[(os.path.basename(documents_path).strip().replace('.','_').replace(':','_').replace(' ','_'),text_vec)]
        return text_vec

    def search_index(self, query):
        output = []
        """比较相似度，并输出前output_num个最相似的文本"""
        conn = sqlite3.connect(self.database_path) #创建连接
        input_vec = self.encode(query)
        #获取表名   
        cursor = conn.cursor() #创建游标
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ({})".format(
            ','.join(['?']*len(self.tables))
            ), self.tables)
        table_names = cursor.fetchall()
        # 遍历每个表，进行相似度搜索
        for table_name in table_names:
            table_name = table_name[0]
            print(colored(f"正在搜索表{table_name}", "blue"))
            cursor.execute(f'SELECT content, embedding FROM "{table_name}"')
            database = cursor.fetchall()
            for row in database:
                content = row[0]
                embedding = pickle.loads(row[1])
                cos = self.cal_cos(input_vec, embedding)
                output.append((cos,table_name,content))
        conn.close()
        # 根据相似度排序并返回前top_n个结果
        output.sort(key=lambda x: x[0], reverse=True)
        
        if len(output) < self.top_n:
            return output
        else:
            return output[:self.top_n]

    def check_table_exist(self, table_name):
        # 检查表是否存在
        conn = sqlite3.connect(self.database_path) #创建连接
        cursor = conn.cursor() #创建游标
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if cursor.fetchone(): #fetchone返回的是一个元组，如果为空，则返回None
            return True
        return False
    
    def save_index(self, text_vec):
        if text_vec is None:
            print(colored("没有有效的文件", "red"))
            return
        
        print(colored(f"正在保存向量索引到文件{self.database_path}", "green"))
        conn = sqlite3.connect(self.database_path)  # 创建连接
        cursor = conn.cursor()  # 创建游标

        for table_name, vectors in text_vec:
            # 将文件路径作为表名，并替换掉不适合的符号
            table_name = f'"{table_name}"'
            # 创建表：id, content, embedding（如果不存在则创建）
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    embedding BLOB
                )
            ''')
            # 对于每个表，插入数据
            for vec in vectors:
                # 将 NumPy 数组转换为二进制格式（BLOB）
                embedding_blob = sqlite3.Binary(pickle.dumps(vec["embedding"]))  # 使用 pickle 序列化
                cursor.execute(f'''
                    INSERT INTO {table_name} (content, embedding)
                    VALUES (?, ?)
                ''', (vec["content"], embedding_blob))
                print(colored(f"文件{table_name}保存成功", "green"))
        conn.commit()
        conn.close()

    def load_index(self):
        # 用户选择路径
        from tkinter.filedialog import askopenfilename
        # 选择文档路径,可以是文件夹，也可以是pdf，txt，md，docx文件
        from tkinter.filedialog import askdirectory
        from tkinter import messagebox
        selection_type = messagebox.askquestion("选择类型", "您要选择文件/文件夹？(是/否)", icon='question')
        if selection_type == 'yes': 
            documents_path = askopenfilename(title="选择文件",
                                         filetypes=[("PDF文件", "*.pdf"), 
                                                    ("文本文件", "*.txt"), 
                                                    ("Markdown文件", "*.md"), 
                                                    ("Docx文件", "*.docx")])
        else:
            documents_path = askdirectory(title="选择文件夹")
        text_vec = self.create_index(documents_path)
        #保存到文件
        self.save_index(text_vec)


    def delete_table(self):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name!='sqlite_sequence'")
        tables = [row[0] for row in cursor.fetchall()]
        if not tables:
            print(colored("没有找到任何文献。", "yellow"))  
            return
        # 显示表名列表
        print("-"*50)
        print(colored("数据库中已有的文献:", "blue"))
        print("-"*50)
        for idx, table in enumerate(tables, start=1):
            print(colored(f"{idx}. {table}", "blue"))
        print("-"*50)
        # 让用户选择序号，分号;隔开，或者:表示连续 如1:5;7;9表示1,2,3,4,5,7,9
        user_input = input(colored("请选择你要删除的文献的编号（例如1:5;7;9;all）: ", "cyan"))
        selected_indices = set()
        #ALL表示所有表
        if user_input.strip().lower() == 'all':
            selected_indices = set(range(1, len(tables)+1))
        else:
            for part in user_input.split(';'):
                if ':' in part:
                    start, end = part.split(':')
                    selected_indices.update(range(int(start), int(end)+1))
                else:
                    selected_indices.add(int(part))
            # 过滤无效的索引    
            selected_indices = {i for i in selected_indices if 1 <= i <= len(tables)}
            selected_tables = [tables[i-1] for i in sorted(selected_indices)]
        print("-"*50)
        print(colored("已选择要删除的文献:", "blue"))
        for idx,table in enumerate(selected_tables, start=1):
            print(colored(f"{idx}. {table}", "green"))
        print("-"*50)
        #询问是否删除
        print(colored("是否删除选中的文献？(y/n)", "red"))
        choice = input().strip()
        if choice == 'y':
            # 删除选中的表
            for table_name in selected_tables:
                cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                print(colored(f"已删除文献{table_name}", "green"))
            conn.commit()
            conn.close()
        else:
            print(colored("已取消删除", "red"))

    def show_table_info(self):
        conn = sqlite3.connect(self.database_path) #创建连接
        cursor = conn.cursor() #创建游标
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table'")
        table_names = cursor.fetchall() #fetchall返回的是一个元组列表，每个元组包含一个表名
        conn.close()
        print(colored(table_names, "blue"))
        for table_name in table_names:
            if table_name[0] == "sqlite_sequence":
                continue
            table_name = table_name[0]
            print(colored(f"表名：{table_name}", "blue"))
            #检查表的行数
            conn = sqlite3.connect(self.database_path) #创建连接
            cursor = conn.cursor() #创建游标
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            row_count = cursor.fetchone()[0]
            print("表的行数：",row_count)
            #检查表的列数
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            column_names = [column[1] for column in cursor.fetchall()]
            print("表的列数：",len(column_names))
            #提取第一行的第一列、第二列、第三列
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 1')
            row = cursor.fetchone()
            print("第一行的第一列：",row[0])
            print("第一行的第二列：",row[1])
            conn.close()

if __name__ == "__main__":
    database_path = os.getenv("DATABASE_PATH")
    vector_indexer = VectorIndexer(database_path=database_path)
    vector_indexer.load_index()
    vector_indexer.show_table_info()
    #output=vector_indexer.search_index("你好")
    #print(output)
    

        

