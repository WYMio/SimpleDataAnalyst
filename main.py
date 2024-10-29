import pandas as pd
import streamlit as st
from csv_qa import dataframe_agent, initial_insight


def create_chart(input_data, chart_type):
    df_data = pd.DataFrame(input_data["data"], columns=input_data["columns"])
    df_data.set_index(input_data["columns"][0], inplace=True)
    if chart_type == "bar":
        st.bar_chart(df_data)
    elif chart_type == "line":
        st.line_chart(df_data)
    elif chart_type == "scatter":
        st.scatter_chart(df_data)


def create_map(input_data):
    df = pd.DataFrame(input_data)
    st.map(df)


st.title("💡 CSV数据分析Agent")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="在此输入您的 OpenAI API Key")
    st.markdown("<span style='color: gray; font-size: 12px;'>不会保存此密钥，您输入的密钥仅在当前会话中可用。</span>",
                unsafe_allow_html=True)
    openai_api_base = st.text_input("API Base", placeholder="在此输入您的API Base")
    # 选择要使用的模型
    model = st.selectbox("请选择您要使用的模型：",
                         ["gpt-4o-mini", "gpt-4o"])

    st.markdown("<br><br>", unsafe_allow_html=True)  # 使用HTML的换行标签

    with st.expander("点击展开使用说明", expanded=False):
        # 使用HTML样式设置小字体和灰色
        st.markdown("""
        <style>
            .small-text {
                font-size: 13px;
                color: gray;
            }
        </style>
        """, unsafe_allow_html=True)
        # 应用样式
        st.markdown("""<div class="small-text">
        这是一个数据分析Agent。它可以读取您上传的csv格式的数据集，回答有关数据集的问题、提取数据或进行可视化，
        Agent会根据您的需求回答问题或绘制图表（支持表格、散点图、折线图、条形图），如果有经纬度信息，也支持绘制地图。<br><br>
        以下是一些可能会提出的问题：<br>
        - 男性用户与女性用户的平均消费金额分别为多少？<br>
        - 请提取出所有年龄大于30岁的用户数据。<br>
        - 请绘制用户职业的条形图。<br>
        - 用户年龄与使用时长之间有什么关系？<br><br>
        注意：<br>
        - 请先使用"gpt-4o-mini"模型，如果出现报错，再尝试选择"gpt-4o"模型(生成散点图与地图推荐使用gpt-4o模型)。<br>
        - 上传的csv文件数据不宜过多，否则处理的时间会太长，提取数据时也容易超过LLM的上下文窗口。建议csv文件不要超过100条数据。<br>
        - 该Agent使用Pandas库对DataFrame进行处理，超出Pandas库能力范围的问题可能会报错。<br>
        - 如要绘制散点图或地图，上传的数据请不要超过50条。<br><br><br></div>
        """, unsafe_allow_html=True)


st.subheader("上传数据文件")

# 上传csv文件
data = st.file_uploader("请在这里上传您的数据文件（CSV格式）：", type="csv")
# 展示原始数据
if data:
    st.session_state.df = pd.read_csv(data)
    with st.expander("原始数据"):
        st.dataframe(st.session_state.df)

st.markdown("---")


st.subheader("数据初步洞察")
st.write("如果您不熟悉这个数据集，可以点击按钮自动生成对该数据集的初步洞察，这可能会为您后续想询问的问题提供启发和方向。")

# 问题输入框和生成回答按钮
about_dataset = st.text_area("请输入关于该数据集的简介（非必填）:")
button_con = st.button("生成数据洞察")

# 运行一次数据洞察
if button_con and not openai_api_key:
    st.info("请输入您的OpenAI API密钥")
if button_con and "df" not in st.session_state:
    st.info("请先上传数据文件")
if button_con and openai_api_key and "df" in st.session_state:
    with st.spinner("AI正在思考中，请稍等..."):
        pre_conclusions = initial_insight(model, openai_api_key, openai_api_base, st.session_state["df"], about_dataset)
        st.session_state.con = pre_conclusions

if "con" in st.session_state:
    with st.expander("点击展开初步数据洞察结果", expanded=False):
        st.markdown(st.session_state.con)

st.markdown("---")


st.subheader("输入问题")

# 问题输入框和生成回答按钮
query = st.text_area("请输入您关于以上表格的问题，数据提取请求，或可视化要求（支持散点图、折线图、条形图）:")
button = st.button("生成回答")

# 运行一次问答
if button and not openai_api_key:
    st.info("请输入您的OpenAI API密钥")
if button and "df" not in st.session_state:
    st.info("请先上传数据文件")
if button and not query:
    st.info("请输入您想查询的问题")
if button and openai_api_key and "df" in st.session_state and query:
    with st.spinner("AI正在思考中，请稍等..."):
        response_dict = dataframe_agent(model, openai_api_key, openai_api_base, st.session_state["df"], query)
        if "answer" in response_dict:
            st.write(response_dict["answer"])
        if "table" in response_dict:
            st.table(pd.DataFrame(response_dict["table"]["data"],
                                  columns=response_dict["table"]["columns"]))
        if "bar" in response_dict:
            create_chart(response_dict["bar"], "bar")
        if "line" in response_dict:
            create_chart(response_dict["line"], "line")
        if "scatter" in response_dict:
            create_chart(response_dict["scatter"], "scatter")
        if "map" in response_dict:
            create_map(response_dict["map"])
