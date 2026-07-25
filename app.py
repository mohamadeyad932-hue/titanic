import gradio as gr
import joblib
import pandas as pd
from fastapi import FastAPI

# تحميل النموذج
model = joblib.load("titanic_model.joblib")

def predict_survival(pclass, sex, age, sibsp, parch, fare):
    # تجهيز قيم الجنس بناءً على التكويد الجغرافي One-Hot Encoding
    sex_female = 1 if sex == "female" else 0
    sex_male = 1 if sex == "male" else 0
    
    # بناء صف البيانات بنفس أسماء الأعمدة المتوقعة من التدريب
    input_dict = {
        'PassengerId': 0,
        'Pclass': int(pclass),
        'Age': float(age),
        'SibSp': int(sibsp),
        'Parch': int(parch),
        'Fare': float(fare),
        'Name': 0,
        'Ticket': 0,
        'Sex_female': sex_female,
        'Sex_male': sex_male,
        'Embarked_C': 0,
        'Embarked_Q': 0,
        'Embarked_S': 1  # افتراضي Port S
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # ترتيب الأعمدة تلقائياً إذا كان النموذج يحفظ الترتيب الأصلي
    try:
        expected_cols = model.feature_names_in_
        input_df = input_df[expected_cols]
    except AttributeError:
        pass

    # التنبؤ بالنتيجة والاحتمالية
    prediction = model.predict(input_df)[0]
    
    try:
        probability = model.predict_proba(input_df)[0][1]
        prob_text = f" (نسبة النجاة المتوقعة: {probability*100:.1f}%)"
    except Exception:
        prob_text = ""
    
    if prediction == 1:
        return f"مبروك! التوقع: نجاة 🎉{prob_text}"
    else:
        return f"للأسف، التوقع: عدم النجاة 🏊‍♂️{prob_text}"

# إنشاء تطبيق FastAPI
app = FastAPI()

# إنشاء واجهة Gradio
io = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Dropdown([1, 2, 3], value=3, label="(Pclass) درجة التذكرة"),
        gr.Radio(["male", "female"], value="male", label="(Sex) الجنس"),
        gr.Slider(0, 100, value=25, label="(Age) العمر"),
        gr.Number(label="(SibSp) عدد الأخوة/الأزواج معك", value=0),
        gr.Number(label="(Parch) عدد الآباء/الأطفال معك", value=0),
        gr.Number(label="(Fare) سعر التذكرة", value=10.5)
    ],
    outputs="text",
    title="🚢 التنبؤ بالنجاة من تايتنك",
    description="نموذج ذكاء اصطناعي (Random Forest) للتنبؤ بالنجاة"
)

# دمج واجهة Gradio داخل تطبيق FastAPI
app = gr.mount_gradio_app(app, io, path="/")
