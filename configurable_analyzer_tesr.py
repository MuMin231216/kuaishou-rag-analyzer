import csv
import dashscope
import os

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


def read_csv_data(filename):
    earnings = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            earnings.append(float(row['earning']))
    return earnings


def build_prompt(data, target, mode, dimensions):
    # 根据 dimensions 动态构建任务列表
    tasks = []
    if "low_days" in dimensions:
        tasks.append("列出低于目标的天数和数值")
    if "avg" in dimensions:
        tasks.append("计算平均收益")
    if "suggestions" in dimensions:
        tasks.append("给出3条可执行的提升建议")

    task_text = "、".join(tasks)

    if mode == "zero_shot":
        return f"你是快手收益分析专家。数据：{data}，目标：{target}元。请完成任务：{task_text}。"

    elif mode == "few_shot":
        return f"""严格按以下格式输出，不要添加任何其他内容。
示例输出：
低于目标的天数：第3天（70元）
平均收益：81.7元
建议：第3天偏低，建议更换内容类型。

真实数据：{data}，目标：{target}元。
请按相同格式输出，任务包括：{task_text}。"""

    elif mode == "cot":
        return f"""按步骤思考后给出最终结论。
数据：{data}，目标：{target}元。
任务：{task_text}
请输出【推理过程】和【最终结论】。"""

    else:
        raise ValueError(f"未知模式: {mode}")


def analyze_earning(data, target, mode, dimensions):
    prompt = build_prompt(data, target, mode, dimensions)
    response = dashscope.Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.output.text


# ===== 主程序 =====
earnings = read_csv_data('E:/project/my_csv_file/Kuaishou Earning1.csv')

# 在这里切换 mode 测试
config = {
    "mode": "cot",  # 改成 "zero_shot" 或 "few_shot" 试试
    "target": 80,
    "dimensions": ["low_days", "avg", "suggestions"]
}

result = analyze_earning(
    earnings,
    config["target"],
    config["mode"],
    config["dimensions"]
)
print(result)