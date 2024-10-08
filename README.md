# 文献分析助手

## 项目简介

**文献分析助手** 是一款基于向量索引和人工智能的高级文献分析工具。它能够帮助用户高效地管理、检索和分析大量文献资料，通过自然语言交互为用户提供专业、准确的回答。该工具集成了OpenAI的嵌入模型，支持多种文档格式，并提供便捷的命令行界面，适用于研究人员、学生以及需要处理大量文献的专业人士。

## 功能

- **文献索引与管理**：支持PDF、TXT、Markdown和DOCX等多种文档格式的文本提取与向量化，自动创建和管理向量索引库。
- **智能问答**：基于向量数据库和OpenAI模型，能够理解用户问题并在相关文献中检索答案。
- **聊天记录管理**：支持保存、加载和清除聊天记录，便于用户跟踪和回顾对话历史。
- **用户友好的命令行界面**：通过简单的命令操作，实现创建知识库、进行问答、管理聊天记录等功能。
- **知识库管理**：支持创建、加载、保存和删除知识库，便于用户管理和切换不同的知识库。

## 安装

### 前提条件

- **Python 3.10** 及以上版本

### 安装步骤

1. **克隆仓库**

   ```bash
   git clone https://github.com/Code-WSY/Paper_Copilot.git
   cd Paper_Copilot
   ```

2. **创建虚拟环境（可选）**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix系统
   venv\Scripts\activate     # Windows系统
   ```

3. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

4. **配置环境变量**

   在项目根目录下创建一个 `.env` 文件，并添加以下内容：

   ```env
   API_KEY=your_openai_api_key
   BASE_URL=your_openai_base_url
   DATABASE_PATH=path_to_your_database.db
   ```

   - `API_KEY`：你的OpenAI API密钥。
   - `BASE_URL`：提供OpenAI服务的URL。
   - `DATABASE_PATH`：向量索引数据库的存储路径。

## 使用方法

1. **启动程序**

   在命令行中执行：

   ```bash
   python main.py
   ```

2. **命令列表**

   启动后，你将看到以下可用命令：

   ```
   /create                创建知识库
   /chat <问题>            基于知识库进行问答
   /save_chat_history     保存聊天记录
   /clear_chat_history    清除聊天记录
   /load_chat_history     加载聊天记录
   /save_last_response    保存上一次的回答为Markdown文件
   /help                  显示帮助信息
   /quit                  退出程序
   ```

3. **创建知识库**

   使用 `/create` 命令，程序将引导你选择要索引的文档或文件夹，自动提取文本并创建向量索引。

4. **进行问答**

   使用 `/chat` 命令后跟你的问题，例如：

   ```
   /chat 这篇论文的创新点是什么？
   ```

   系统将基于知识库提供专业的回答。

5. **管理聊天记录**

   - **保存聊天记录**

     ```
     /save_chat_history
     ```

   - **加载聊天记录**

     ```
     /load_chat_history
     ```

   - **清除聊天记录**

     ```
     /clear_chat_history
     ```

6. **保存回答**

   使用 `/save_last_response` 命令将上一次的回答保存为Markdown文件。

7. **获取帮助**

   使用 `/help` 命令查看所有可用命令的说明。

8. **退出程序**

   使用 `/quit` 命令退出程序。

## 环境变量

请在 `.env` 文件中配置以下环境变量：

- `API_KEY`：你的OpenAI API密钥。
- `BASE_URL`：OpenAI服务的基础URL。
- `DATABASE_PATH`：向量索引数据库的存储路径。

示例 `.env` 文件：

```env
API_KEY=sk-your_openai_api_key
BASE_URL=https://api.openai.com/v1
DATABASE_PATH=./data/vector_index.db
```

## 许可证

本项目采用 [MIT 许可证](LICENSE)。