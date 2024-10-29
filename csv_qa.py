import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


PROMPT_TEMPLATE = """
你是一位数据分析助手，你的回应内容取决于用户的请求内容。

1. 对于文字回答的问题，按照这样的格式回答：
   {"answer": "<你的答案写在这里>"}
例如：
   {"answer": "订单量最高的产品ID是'MNWC3-067'"}

2. 如果用户需要一个表格，按照这样的格式回答：
   {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

3. 如果用户的请求适合返回条形图，按照这样的格式回答：
   {"bar": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

4. 如果用户的请求适合返回折线图，按照这样的格式回答：
   {"line": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

5. 如果用户的请求适合返回散点图，按照这样的格式回答：
   {"scatter": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}
注意：我们只支持三种类型的图表："bar", "line" 和 "scatter"。

6. 如果用户的请求适合用地图展示，按照这样的格式返回经纬度信息：
    {"map": {"lat": [37.76, 37.77, 37.78], "lon": [-122.43, -122.44, -122.45]}}


请将输出以JSON格式返回，不允许出现任何其他符号，例如```json。请注意要将"columns"列表和数据列表中的所有字符串都用双引号包围。
例如：{"columns": ["Products", "Orders"], "data": [["32085Lip", 245], ["76439Eye", 178]]}

你要处理的用户请求如下： 
"""


INSIGHT_PROMPT = """
你是一位可以根据数据集的基本信息进行初步分析的专家。你将收到一个包含数据集基本信息的字典，该字典包括键‘columns’, ‘shape’,
‘description’, ‘missing_values’。首先你需要对这个数据集的基本情况进行介绍，包括数据集有多少特征（变量），分别是什么，
有多少条记录（样本），数据是否有缺失。然后你需要充分利用接收到的基本信息得出关于该数据集的初步结论，例如参考'description'
中的内容得出一般性结论，这将为稍后更具体的数据分析提供灵感和方向。请使用与用户输入相同的语言回答。
以下是数据集的基本信息：
"""


# 传入DataFrame并分析数据
def analyze_data(df):
    summary = {
        "columns": df.columns.tolist(),
        "shape": df.shape,
        "description": df.describe(),
        "missing_values": df.isnull().sum().to_dict()
    }
    return summary


# 通过LLM得出初步分析结论
def initial_insight(model, openai_api_key, openai_api_base, df, about_dataset=None):
    # 对上传的数据集进行初步分析，并生成结论

    # 创建prompt
    summary_str = str(analyze_data(df)).replace('{', '{{').replace('}', '}}')

    prompt = ChatPromptTemplate.from_messages([
        ("system", INSIGHT_PROMPT),
        ("human", "数据集简介：\n{user_input}\n\n数据集基本信息：\n"+summary_str)
    ])

    # 创建llm并组成chain
    model = ChatOpenAI(model=model, openai_api_key=openai_api_key, openai_api_base=openai_api_base)
    chain = prompt | model
    # 执行chain
    response = chain.invoke({"user_input": about_dataset})
    return response.content


def dataframe_agent(model, openai_api_key, openai_api_base, df, query):
    # 对用户提出的关于数据集的具体问题进行解答
    model = ChatOpenAI(model=model, temperature=0,
                       openai_api_key=openai_api_key, openai_api_base=openai_api_base)
    agent = create_pandas_dataframe_agent(llm=model,
                                          df=df,
                                          allow_dangerous_code=True,
                                          agent_executor_kwargs={"handle_parsing_errors": True},
                                          verbose=True)
    prompt = PROMPT_TEMPLATE + query
    response = agent.invoke({"input": prompt})
    response_dict = json.loads(response["output"])
    return response_dict
