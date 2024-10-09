from src.agent import Agent
from src.vector_indexer import VectorIndexer
from termcolor import colored
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit import prompt
import os

def start():
    print(colored("命令:", "cyan"))
    print(colored("-"*50, "cyan"))
    commands = {
        "/chat": "基于知识库问答（后接问题）",
        "/create": "创建知识库",
        "/delete": "删除数据库中的文献",
        "/select": "重新选择文献",
        "/help": "显示帮助信息",
        "/quit": "退出程序",
        "/save": "保存聊天记录",
        "/clear": "清除聊天记录",
        "/load": "加载聊天记录",
        "/last_md": "保存上一次的回答为markdown文件",

    }
    command_list = list(commands.keys())
    for cmd, desc in commands.items():
        print(f"{colored(cmd, 'magenta'):<10} {colored(desc, 'dark_grey')}")
    completer = WordCompleter(
            command_list)
    #初始化agent
    agent = Agent(prompt_path="src/prompt/文献分析助手.md")
    while True:
        #限制聊天记录长度
        if len(agent.history) > 20:
            #只保留最后20条记录
            agent.history = agent.history[-20:]
        print(colored("You:\n", "cyan"))
        command = prompt(completer=completer).strip()
        if command.startswith("/quit"):
            break
        elif command.startswith("/last_md"):
            agent.save_last_response()
        elif command.startswith("/create"):
            #操作数据库
            database_path = os.getenv("DATABASE_PATH")
            #初始化向量数据库
            vector_indexer = VectorIndexer(database_path=database_path)
            vector_indexer.load_index()
            #vector_indexer.show_table_info()
        elif command.startswith("/save"):
            agent.save_chat_history()
        elif command.startswith("/clear"):
            agent.clear_chat_history()
        elif command.startswith("/load"):
            agent.load_chat_history()
        elif command.startswith("/chat"):
            agent.chat_with_vector_database(command[6:])
        elif command.startswith("/help"):
            print(colored("命令:", "cyan"))
            print(colored("-"*50, "cyan"))
            for cmd, desc in commands.items():
                print(f"{colored(cmd, 'magenta'):<10} {colored(desc, 'dark_grey')}")
        elif command.startswith("/delete"):
            #删除文献
            agent.delete_table()
        elif command.startswith("/select"):
            #选择文献
            agent.select_tables()
        else:
            #不加载知识库，直接问答
            agent.chat_with_ai(command)
if __name__ == "__main__":
    start()

