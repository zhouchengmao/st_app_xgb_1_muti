import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import numpy as np
import joblib


flag = True


def check_invalid_input_val(val):
    return val is None or str(val).strip() == "" or str(val).strip().lower().startswith("please")


def setup_ml_model():
    global flag
    if flag:
        try:
            model = joblib.load('mxgb_muti.pkl')
            print("成功从文件中加载模型。")
            return model
        except FileNotFoundError:
            print("未找到模型文件，开始重新训练模型。")
            flag = False

    data = pd.read_csv('pocd_muti.csv')
    X = data[['ES', 'NYHA Class', 'PAH', 'SP02', 'ASA PS']]
    y = data['Maternal combined endpoint']

    model = xgb.XGBClassifier()
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    with st.spinner('正在进行模型训练...'):
        for train_index, val_index in kfold.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = np.mean(y_pred == y_val)
            cv_scores.append(accuracy)
    avg_accuracy = np.mean(cv_scores)
    print(f"5折交叉验证的平均准确率: {avg_accuracy}")
    model.fit(X, y)

    # 保存模型到文件
    joblib.dump(model, 'mxgb_muti.pkl')
    print("模型已保存到文件 mxgb_muti.pkl")

    return model


def render_ui(model):
    col1, col2 = st.columns(2)

    with col1:
        # 肺动脉压（PAH）输入框
        pah = st.number_input("肺动脉压（mmHg）", min_value=0, value=None, placeholder="Please Input PAH")
        # 脉搏氧饱和度（SPO2）输入框
        spo2 = st.number_input("脉搏氧饱和度（%）", min_value=0, max_value=100, value=None, placeholder="Please Input SPO2")
        # 艾森曼格综合征（ES）输入框
        es = st.selectbox("艾森曼格综合征（ES）", ["Please Select ES", "0", "1"], index=0)

    with col2:
        # ASA分级输入框
        asa_options = ["Please Select ASA", "1", "2", "3"]
        asa = st.selectbox("ASA分级", asa_options, index=0)
        # NYHA分级输入框
        nyha_options = ["Please Select NYHA", "1", "2", "3"]
        nyha = st.selectbox("NYHA分级", nyha_options, index=0)

    result_col, button_col = st.columns([3, 1])

    with result_col:
        result_placeholder = st.empty()

    if button_col.button("开始计算"):
        flag = True
        for v in [es, nyha, pah, spo2, asa]:
            if check_invalid_input_val(v):
                st.warning("请完整填写所有输入项！")
                flag = False
                break

        if flag:
            input_data = pd.DataFrame({
                'ES': [int(es)],
                'NYHA Class': [int(nyha)],
                'PAH': [pah],
                'SP02': [spo2],
                'ASA PS': [int(asa)]
            })
            prediction = model.predict(input_data)
            result_text = f"产妇复合终点事件: {prediction[0]}"
            result_col.markdown(f"<p style='text-align: right; margin-top: 8px;'>{result_text}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    st.image("logo.jpg", width=300, use_column_width='never')
    st.markdown(
        """
        <div style="
            height: 2px;
            border: none;
            border-radius: 5px;
            background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
        " />
        """,
        unsafe_allow_html=True
    )
    st.title("围术期产妇复合终点事件风险预测模型——预测工具（基于XGBoost）")
    model = setup_ml_model()
    render_ui(model)
