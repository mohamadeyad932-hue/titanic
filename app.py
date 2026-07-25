import gradio as gr
import joblib
import pandas as pd
from fastapi import FastAPI

# تحميل النموذج
model = joblib.load("titanic_model.joblib")

def predict_survival(pclass, sex, age, sibsp, parch, fare):
    sex_num = 1 if sex == "female" else 0
    input_data = pd.DataFrame([{
        'Pclass': int(pclass),
        'Sex': sex_num,
        'Age': float(age),
        'SibSp': int(sibsp),
        'Parch': int(parch),
        'Fare': float(fare)
    }])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        return f"مبروك! نسبة النجاة عالية: {probability*100:.1f}% 🎉"
    else:
        return f"للأسف، الفرص ضعيفة.. نسبة النجاة: {probability*100:.1f}% 🏊‍♂️"

# إنشاء تطبيق FastAPI
app = FastAPI()

# إنشاء واجهة Gradio
io = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Dropdown([1, 2, 3], value=3, label="درجة التذكرة (Pclass)"),
        gr.Radio(["male", "female"], value="male", label="الجنس (Sex)"),
        gr.Slider(0, 100, value=25, label="العمر (Age)"),
        gr.Number(label="عدد الأخوة/الأزواج معك (SibSp)", value=0),
        gr.Number(label="عدد الآباء/الأطفال معك (Parch)", value=0),
        gr.Number(label="سعر التذكرة (Fare)", value=10.5)
    ],
    outputs="text",
    title="🚢 التنبؤ بالنجاة من تايتنك",
    description="نموذج Random Forest للتنبؤ بالنجاة"
)

# دمج واجهة Gradio داخل تطبيق FastAPI
app = gr.mount_gradio_app(app, io, path="/")