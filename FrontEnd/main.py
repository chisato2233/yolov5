import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import subprocess
import uuid
import torch
import numpy as np
import torch
from torchvision import transforms

from MyColor import color_detect
import linear.LinearRegression.MultivariateLinearRegression as ml
import sys;
def extract_total_frames(log_line):
    # 假设日志格式为 "video 1/1 (1/197) ..."
    # 我们需要提取 '197' 这个数字
    parts = log_line.split()
    for part in parts:
        if '/' in part:
            total_frames = part.split('/')[1].strip('):')
            return int(total_frames)
    return None

st.set_page_config(page_title="大桃分析", layout="wide")
st.markdown("<style>.big-font {font-size:20px !important;}</style>", unsafe_allow_html=True)

# 在侧边栏添加选择功能的选项

st.sidebar.markdown("# 大桃 🍑 分析 ")
# st.sidebar.markdown("""
#     <style>
#     .indent-text {
#         text-indent: 2em;  /* 控制首行缩进的大小 */
#         margin-bottom: 20px;
#     }
#     </style>
#     <div class="indent-text">本网站可提供“着色度分析”、“品种识别”、“甜度分析”三个功能。</div>
#     """, unsafe_allow_html=True)



st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
    <div class="title">大桃 🍑 分析</div>
    """, unsafe_allow_html=True)


option = st.sidebar.selectbox(
    """
    请选择一个功能 😊😊：
        
    """,
    ('🎨着色度分析', '👁️品种识别', '🍬甜度分析')
)

# 根据选择的功能显示不同的信息和上传控件
if option == '🎨着色度分析':
    
    st.markdown("## 🎨着色度分析")
    st.write('  着色度分析是指对桃子的特征颜色进行提取并绘制轮廓图，用以分析桃子的成熟度。'
             '在下方上传需要分析的桃子图片即可自动分析并生成分析结果。')
    
    
    
    uploaded_file = st.file_uploader("请上传要分析的桃子的图片 ： ", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        orginal_image = Image.open(uploaded_file)
        st.image(orginal_image, caption='上传的图片', use_column_width=True)
        
        if st.button("识别！😆"):
            with st.spinner("AI正在疯狂思考... 🤯🤯🤯"):
                st.image(color_detect(uploaded_file), caption='分析结果', use_column_width=True)

elif option == '👁️品种识别':

    # 标题和描述
    st.markdown("## 👁️品种识别")
    st.write('品种识别是指使用依托于自主构建的包含主要桃子品种在不同成熟阶段、不同光照条件下的数据集，'
            '训练并优化得出的桃子品种识别模型精准快速地识别桃子品种。在下方上传需要分析的桃子图片或视频即可自动识别并生成识别结果。')

    # 文件上传
    uploaded_file = st.file_uploader("请上传要分析的桃子的图片或视频：", type=["jpg",  "png"])
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()  # 获取文件扩展名

        # 保存文件到临时目录
        unique_filename = f"{uuid.uuid4()}.{file_type}"
        if file_type in ["jpg", "png"]:
            # 处理图片
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image_path = os.path.join("temp_images", unique_filename)
            os.makedirs("temp_images", exist_ok=True)
            image.save(image_path)
        else:
            # 处理视频
            video_path = os.path.join("temp_videos", unique_filename)
            os.makedirs("temp_videos", exist_ok=True)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

        if st.button("开始识别！😆"):
            with st.spinner("模型在检测... 🤯🤯🤯"):
                if file_type in ["jpg", "png"]:
                    # 图片识别逻辑
                    result = subprocess.run([
                        sys.executable, "detect.py",
                        "--weights", "best.pt",
                        "--source", image_path,
                        "--project", "temp_results",
                        "--name", "detect_result",
                        "--exist-ok",
                        "--line-thickness", "2",
                    ], check=True)

                    if result.returncode != 0:
                        st.error("错误输出: " + result.stderr.decode())

                    # 读取并显示结果图像
                    result_path = os.path.join("temp_results", "detect_result", unique_filename)
                    result_image = Image.open(result_path)
                    st.image(result_image, caption="Detection Result", use_column_width=True)
                else:
                    # 视频识别逻辑
                    # 运行视频处理命令
                    process = subprocess.Popen(
                        [
                            sys.executable, "detect.py",
                            "--weights", "best.pt",
                            "--source", video_path,
                            "--project", "temp_results",
                            "--name", "detect_result",
                            "--exist-ok",
                            "--line-thickness", "2",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                    # 初始化进度条
                    progress_bar = st.progress(0)
                    frame_count = 0

                    # 读取输出，更新进度条
                    for line in process.stdout:
                        if "video 1/1" in line:
                            frame_count += 1
                            progress = frame_count / 197  # 临时假设总帧数
                            progress_bar.progress(progress)

                    # 确保进程完成
                    process.communicate()

                    # 显示结果视频
                    result_video_path = os.path.join("temp_results", "detect_result", unique_filename)
                    st.video(result_video_path)

                    # 结束处理后完成进度条
                    progress_bar.progress(1.0)
                    st.success("识别完成！")

                    # 清理临时生成的文件的按钮
                if st.button('清理临时文件'):
                    os.remove(os.path.join("temp_images" if file_type in ["jpg", "jpeg", "png"] else "temp_videos", unique_filename))
                    os.remove(os.path.join("temp_results", "detect_result", unique_filename))
                    st.success("临时文件已清理！")

elif option == '🍬甜度分析':
    st.markdown("## 🍬甜度分析")
    st.write('这里是甜度分析的功能介绍和使用说明。')

    uploaded_file = st.file_uploader("上传 txt 文件", type=["txt"])
    if uploaded_file is not None:
        # 读取 txt 文件内容，并转换为特征值数组
        content = uploaded_file.read().decode("utf-8")
        features = np.fromstring(content, sep=' ')
        features = features.reshape(1, -1)

        # 当用户上传文件后直接预测，不需要额外的提交按钮
        prediction = ml.predict_sugar_content(features)  # 调用模型进行预测
        
        st.write(f'预测的甜度值为：{prediction}%')
        

