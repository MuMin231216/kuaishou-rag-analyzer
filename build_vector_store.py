import os
import dashscope
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

# 设置 API Key（必须）
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 文件夹路径（改成你存放 txt 文件的目录）
folder_path = "E:/project/my_txt_file/kuaishou-earning"

# 读取所有 txt 文件
documents = []
for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(folder_path, file), encoding="utf-8")
        documents.extend(loader.load())

# 切分文档（每段 500 字符，重叠 50 字符）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 创建向量数据库
embeddings = DashScopeEmbeddings(model="text-embedding-v1")
vector_store = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
#vector_store.persist()

print(f"✅ 成功创建向量数据库，共 {len(docs)} 个文档片段")