import os
import csv
import dashscope
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

os.environ["DASHSCOPE_API_KEY"] = "sk-9ad902a79a51469aa15574cdd247e830"
dashscope.api_key = "sk-9ad902a79a51469aa15574cdd247e830"


def read_csv_data(filename):
    earnings = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            earnings.append(float(row['earning']))
    return earnings


def search_knowledge(question):
    embeddings = DashScopeEmbeddings(model="text-embedding-v1")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    results = vector_store.similarity_search(question, k=3)
    return "\n\n".join([doc.page_content for doc in results])


def analyze_with_rag(data, target, user_question):
    # 1. 从知识库检索相关内容
    knowledge = search_knowledge(user_question)

    # 2. 构建提示词
    prompt = f"""
你是快手收益分析专家。

【快手运营知识库】
{knowledge}

【今日收益数据】
{data}
目标：{target}元/天

【用户问题】
{user_question}

请基于知识库内容回答用户问题，不要编造。
"""
    response = dashscope.Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.output.text


earnings = read_csv_data('E:/project/my_csv_file/Kuaishou Earning1.csv')
question = "我的收益为什么这么低？应该几点发视频比较好？"
result = analyze_with_rag(earnings, 80, question)
print(result)