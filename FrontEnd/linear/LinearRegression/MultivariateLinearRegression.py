import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt



from linear.LinearRegression.linear_regression import LinearRegression
"""
平均绝对误差(MAE)  MAE用来衡量预测值与真实值之间的平均绝对误差，MAE越小表示模型越好
"""
def MAE(y, y_pre):
    return np.mean(np.abs(y - y_pre))
"""
均方误差(MSE)   MSE也许是回归中最普通的评价指标，MSE越小表示模型越好
"""
def MSE(y, y_pre):
    return np.mean((y - y_pre) ** 2)
"""
均方根误差(RMSE)   RMSE是在MSE的基础之上开根号而来，RMSE越小表示模型越好
"""
def RMSE(y, y_pre):
    return np.sqrt(MSE(y, y_pre))
"""
平均绝对百分比误差(MAPE)  MAPE和MAE类似，只是在MAE的基础上做了标准化处理，MAPE越小表示模型越好
"""
def MAPE(y, y_pre):
    return np.mean(np.abs((y - y_pre) / y))
"""
R^2评价指标  sklearn在实现线性回归时默认采用了R^2指标，R^2越大表示模型越好
"""
def R2(y, y_pre):
    u = np.sum((y - y_pre) ** 2)
    v = np.sum((y - np.mean(y)) ** 2)
    return 1 - (u / v)

def predict_sugar_content(features_array):
    
    
    data = pd.read_csv('./FrontEnd/linear/data/optical density-sugar degree.csv')

    #信息导入
    train_data = data.sample(frac=0.8)
    test_data = data.drop(train_data.index)

    input_param_name_1 = 'wavelength.180'
    input_param_name_2 = 'wavelength.280'
    input_param_name_3 = 'wavelength.380'
    input_param_name_4 = 'wavelength.480'
    input_param_name_5 = 'wavelength.580'
    input_param_name_6 = 'wavelength.680'
    input_param_name_7 = 'wavelength.780'
    output_param_name = 'sugar'

    x_train = train_data[[input_param_name_1,
                        input_param_name_2,
                        input_param_name_3,
                        input_param_name_4,
                        input_param_name_5,
                        input_param_name_6,
                        input_param_name_7
                        ]].values
    y_train = train_data[[output_param_name]].values

    x_test = test_data[[input_param_name_1,
                        input_param_name_2,
                        input_param_name_3,
                        input_param_name_4,
                        input_param_name_5,
                        input_param_name_6,
                        input_param_name_7
                        ]].values
    y_test = test_data[[output_param_name]].values


    #数据处理
    num_iterations = 500 #迭代次数
    learning_rate = 0.01
    polynomial_degree = 0
    sinusoid_degree = 0

    linear_regression = LinearRegression(x_train, y_train, polynomial_degree, sinusoid_degree)

    (theta, cost_history) = linear_regression.train(
        learning_rate,
        num_iterations
    )
    
    print('开始损失', cost_history[0])
    print('结束损失', cost_history[-1])

    fig, ax = plt.subplots()
    ax.plot(range(num_iterations), cost_history)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Gradient Descent Progress')
    
    # 显示图表
    #st.pyplot(fig)
    y_predictions = linear_regression.predict(x_train)







    # 读取Excel文件
    # 将DataFrame转换为数组
    features = np.array(features_array).reshape(1, -1)
    features_extended = np.append(features, [[1]], axis=1) 
    print(features_extended)
    

    y_result_test = np.dot(features_extended, theta)
    #调整结果格式
    a = np.round(y_result_test * 100, 4)+9.5
    # y_final = [str(val) + "" for val in a]
    # output = " ".join(y_final)
    #输出结果
    with st.expander("查看模型信息"):
        st.write("MAE: ", MAE(y_train, y_predictions))
        st.write("MSE: ", MSE(y_train, y_predictions))
        st.write("MAPE: ", MAPE(y_train, y_predictions))
        st.write("R^2: ", R2(y_train, y_predictions))

        st.write(theta)
        
        st.pyplot(fig)
    return  a[0][0]
