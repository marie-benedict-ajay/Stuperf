import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────
# LABEL MAPPINGS  (display label → internal dataset value)
# ─────────────────────────────────────────────────────
RACE_MAP = {
    'White':  'group A',
    'Black':  'group B',
    'Asian':  'group C',
    'Jew':    'group D',
    'Indian': 'group E',
}
LUNCH_MAP = {
    'Hosteller':   'standard',
    'Day Scholar': 'free/reduced',
}
ATTENDANCE_MAP = {
    'Above 80%': 'completed',
    'Below 80%': 'none',
}
COURSE_MAP = {
    "High School":        "high school",
    "Some High School":   "some high school",
    "Associate's Degree": "associate's degree",
    "Some College":       "some college",
    "Bachelor's Degree":  "bachelor's degree",
    "Master's Degree":    "master's degree",
}

# ─────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0f1117; }

    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a40);
        border: 1px solid #2d3250; border-radius: 12px;
        padding: 20px; text-align: center; margin: 8px 0;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #7c8cf8; }
    .metric-label { font-size: 0.85rem; color: #8b92a5; margin-top: 4px; }

    .pass-badge {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        border: 2px solid #52b788; border-radius: 16px; padding: 30px;
        text-align: center; font-size: 2rem; font-weight: 700; color: #b7e4c7;
    }
    .fail-badge {
        background: linear-gradient(135deg, #4a1520, #6b2737);
        border: 2px solid #e76f51; border-radius: 16px; padding: 30px;
        text-align: center; font-size: 2rem; font-weight: 700; color: #f4a261;
    }
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #e0e0e0;
        border-left: 4px solid #7c8cf8; padding-left: 12px; margin: 24px 0 16px 0;
    }
    .info-box {
        background: #1a1f33; border: 1px solid #2d3250; border-radius: 8px;
        padding: 14px 18px; color: #a0a8c0; font-size: 0.9rem; margin: 8px 0;
    }
    div[data-testid="stSidebar"] { background-color: #13151f; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; font-weight: 600; width: 100%;
    }
    .stButton > button:hover { opacity: 0.85; }
    h1, h2, h3 { color: #e0e0ff !important; }
    .stSelectbox label, .stSlider label, .stTextInput label { color: #a0a8c0 !important; }
    .stTabs [data-baseweb="tab"] { color: #8b92a5; }
    .stTabs [aria-selected="true"] { color: #7c8cf8 !important; border-bottom-color: #7c8cf8 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    np.random.seed(42)
    n = 1000
    genders   = np.random.choice(['male','female'], n, p=[0.48,0.52])
    races     = np.random.choice(['group A','group B','group C','group D','group E'],
                                 n, p=[0.08,0.19,0.32,0.26,0.15])
    par_edu   = np.random.choice(
        ["bachelor's degree","some college","master's degree",
         "associate's degree","high school","some high school"],
        n, p=[0.19,0.23,0.12,0.22,0.13,0.11])
    lunch     = np.random.choice(['standard','free/reduced'], n, p=[0.64,0.36])
    test_prep = np.random.choice(['none','completed'], n, p=[0.64,0.36])

    edu_b={'some high school':0,'high school':3,"associate's degree":5,
           "some college":6,"bachelor's degree":8,"master's degree":10}
    lb_b={'standard':5,'free/reduced':0}; pb_b={'completed':6,'none':0}
    gm={'male':3,'female':-3}; gr={'male':-4,'female':4}; gw={'male':-5,'female':5}
    ms,rs,ws=[],[],[]
    for i in range(n):
        eb=edu_b[par_edu[i]]; lb=lb_b[lunch[i]]; pb=pb_b[test_prep[i]]
        ms.append(int(np.clip(60+eb+lb+pb+gm[genders[i]]+np.random.normal(0,14),0,100)))
        rs.append(int(np.clip(60+eb+lb+pb+gr[genders[i]]+np.random.normal(0,14),0,100)))
        ws.append(int(np.clip(60+eb+lb+pb+gw[genders[i]]+np.random.normal(0,14),0,100)))

    return pd.DataFrame({
        'gender':genders,'race/ethnicity':races,
        'parental level of education':par_edu,
        'lunch':lunch,'test preparation course':test_prep,
        'math score':ms,'reading score':rs,'writing score':ws,
    })


# ─────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────
@st.cache_resource
def train_model(_df):
    df = _df.copy()
    df['average score'] = (df['math score']+df['reading score']+df['writing score'])/3
    df['result'] = (df['average score'] >= 60).astype(int)

    cat_cols = ['gender','race/ethnicity','parental level of education',
                'lunch','test preparation course']
    encoders = {}
    for col in cat_cols:
        encoders[col] = LabelEncoder()
        df[col+'_enc'] = encoders[col].fit_transform(df[col])

    feature_cols = [c+'_enc' for c in cat_cols]+['math score','reading score','writing score']
    X = df[feature_cols]; y = df['result']

    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=100,max_depth=6,random_state=42,n_jobs=-1)
    model.fit(Xtr,ytr)

    y_pred    = model.predict(Xte)
    acc       = accuracy_score(yte,y_pred)
    cv_scores = cross_val_score(model,X,y,cv=5)
    report    = classification_report(yte,y_pred,target_names=['Fail','Pass'],output_dict=True)
    cm        = confusion_matrix(yte,y_pred)
    importances = pd.Series(model.feature_importances_,
                            index=['Gender','Community','Pursuing Course',
                                   'Hosteller/Day Scholar','Attendance',
                                   'Math Score','Reading Score','Writing Score'])
    return model,encoders,Xte,yte,y_pred,acc,cv_scores,report,cm,importances,df


# ─────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Performance")
    st.markdown("**Random Forest Predictor**")
    st.markdown("---")
    st.markdown("### 📁 Dataset")
    uploaded = st.file_uploader("Upload CSV (Kaggle format)", type=['csv'])
    st.markdown("""<div class='info-box'>
    🔗 <b>Dataset:</b><br>
    <a href="https://www.kaggle.com/datasets/spscientist/students-performance-in-exams"
       style="color:#7c8cf8;">Kaggle: Students Performance in Exams</a><br><br>
    Built-in: 1000 students, same format.
    </div>""", unsafe_allow_html=True)
    st.markdown("### ⚙️ Model Info")
    st.markdown("""<div class='info-box'>
    • <b>Algorithm:</b> Random Forest<br>
    • <b>Trees:</b> 100 &nbsp;|&nbsp; <b>Max Depth:</b> 6<br>
    • <b>Pass Threshold:</b> Avg ≥ 60
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# LOAD & TRAIN
# ─────────────────────────────────────────────────────
df_raw = load_and_prepare_data(uploaded)
model,encoders,X_test,y_test,y_pred,acc,cv_scores,report,cm,importances,df = train_model(df_raw)


# ─────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────
st.markdown("# 🎓 Student Performance Prediction")
st.markdown("**Random Forest Classifier** · Kaggle Students Performance in Exams dataset")
st.markdown("---")

c1,c2,c3,c4 = st.columns(4)
for col,val,label in zip([c1,c2,c3,c4],
    [f"{acc:.1%}", f"{cv_scores.mean():.1%}", f"{df['result'].mean():.1%}", str(len(df))],
    ["Model Accuracy","CV Score (5-fold)","Pass Rate","Total Students"]):
    col.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{val}</div>
        <div class='metric-label'>{label}</div></div>""", unsafe_allow_html=True)

st.markdown("---")
tab1,tab2,tab3,tab4 = st.tabs(["🔮 Predict Student","📊 Model Performance","📈 Data Insights","📋 Dataset"])


# ══════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Enter Student Details</div>", unsafe_allow_html=True)
    left, right = st.columns([1,1])

    with left:
        st.markdown("**📌 Student Profile**")
        gender = st.selectbox("Gender", ['male','female'])

        race_display     = st.selectbox("Community", list(RACE_MAP.keys()))
        race_internal    = RACE_MAP[race_display]

        course_display   = st.selectbox("Pursuing Course", list(COURSE_MAP.keys()))
        course_internal  = COURSE_MAP[course_display]

        hostel_display   = st.selectbox("Hosteller / Day Scholar", list(LUNCH_MAP.keys()))
        hostel_internal  = LUNCH_MAP[hostel_display]

        attend_display   = st.selectbox("Attendance", list(ATTENDANCE_MAP.keys()))
        attend_internal  = ATTENDANCE_MAP[attend_display]

    with right:
        st.markdown("**📝 Subject Scores**")
        st.markdown("<div class='info-box'>Rename any subject and adjust its score (0–100).</div>",
                    unsafe_allow_html=True)

        subject_defaults = [
            ("Math",    65),
            ("Reading", 68),
            ("Writing", 67),
            ("Science", 65),
            ("History", 64),
        ]
        subjects, scores = [], []
        for idx, (def_name, def_score) in enumerate(subject_defaults):
            col_name, col_slider = st.columns([1, 2])
            with col_name:
                st.markdown("<br>", unsafe_allow_html=True)
                name = st.text_input("", value=def_name, key=f"sname_{idx}",
                                     placeholder="Subject name")
            with col_slider:
                score = st.slider(f"{def_name} Score", 0, 100, def_score,
                                  key=f"sscore_{idx}")
            subjects.append(name if name.strip() else def_name)
            scores.append(score)

        avg_score = sum(scores)/len(scores)
        st.markdown(f"""<div class='info-box'>
            📊 <b>Average Score:</b> {avg_score:.1f} / 100 &nbsp;|&nbsp;
            {'✅ Above pass threshold (60)' if avg_score >= 60 else '⚠️ Below pass threshold (60)'}
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    if st.button("🔮 Predict Performance"):
        # Map 5 subject scores → 3 model features
        # First 3 subjects go directly; extra subjects blend into an average
        extra_avg = np.mean(scores[3:]) if len(scores) > 3 else np.mean(scores)
        math_s    = int(round((scores[0]*2 + extra_avg)/3))
        reading_s = int(round((scores[1]*2 + extra_avg)/3))
        writing_s = int(round((scores[2]*2 + extra_avg)/3))

        input_data = pd.DataFrame([{
            'gender_enc':
                encoders['gender'].transform([gender])[0],
            'race/ethnicity_enc':
                encoders['race/ethnicity'].transform([race_internal])[0],
            'parental level of education_enc':
                encoders['parental level of education'].transform([course_internal])[0],
            'lunch_enc':
                encoders['lunch'].transform([hostel_internal])[0],
            'test preparation course_enc':
                encoders['test preparation course'].transform([attend_internal])[0],
            'math score':    math_s,
            'reading score': reading_s,
            'writing score': writing_s,
        }])

        pred  = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        st.markdown("---")
        res_col, prob_col = st.columns([1,1])

        with res_col:
            badge_class = 'pass-badge' if pred==1 else 'fail-badge'
            icon = '✅ PASS' if pred==1 else '❌ FAIL'
            msg  = 'predicted to PASS' if pred==1 else 'predicted to FAIL'
            st.markdown(f"""<div class='{badge_class}'>{icon}<br>
                <span style='font-size:1rem;'>Student is {msg}</span></div>""",
                unsafe_allow_html=True)

            st.markdown("<br>**Score Breakdown**", unsafe_allow_html=True)
            for sname, sscore in zip(subjects, scores):
                bar_col = "#52b788" if sscore >= 60 else "#e76f51"
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;margin:5px 0;'>"
                    f"<span style='width:90px;color:#a0a8c0;font-size:.85rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{sname}</span>"
                    f"<div style='flex:1;background:#2d3250;border-radius:4px;height:14px;'>"
                    f"<div style='width:{sscore}%;background:{bar_col};height:14px;border-radius:4px;'></div></div>"
                    f"<span style='color:#e0e0e0;font-size:.85rem;width:30px;text-align:right;'>{sscore}</span>"
                    f"</div>",
                    unsafe_allow_html=True)

        with prob_col:
            fig = go.Figure(go.Bar(
                x=['Fail','Pass'], y=[proba[0]*100, proba[1]*100],
                marker_color=['#e76f51','#52b788'],
                text=[f"{p*100:.1f}%" for p in proba], textposition='outside',
            ))
            fig.update_layout(
                title="Prediction Confidence",
                yaxis_title="Probability (%)", yaxis_range=[0,115],
                plot_bgcolor='#1a1f33', paper_bgcolor='#1a1f33',
                font_color='#e0e0e0', height=300,
                margin=dict(t=40,b=20,l=20,r=20)
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Model Evaluation</div>", unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    for col,label,key in zip([m1,m2,m3,m4],
        ['Precision (Pass)','Recall (Pass)','F1-Score (Pass)','Support (Pass)'],
        ['precision','recall','f1-score','support']):
        val = report['Pass'][key]
        fmt = f"{val:.2f}" if key!='support' else str(int(val))
        col.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='font-size:1.8rem;'>{fmt}</div>
            <div class='metric-label'>{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("")
    perf_col, feat_col = st.columns(2)

    with perf_col:
        st.markdown("**Confusion Matrix**")
        fig,ax = plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor('#1a1f33'); ax.set_facecolor('#1a1f33')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Fail','Pass'], yticklabels=['Fail','Pass'],
                    linewidths=1, linecolor='#2d3250', ax=ax,
                    annot_kws={'color':'white','size':14})
        ax.set_xlabel('Predicted',color='#8b92a5'); ax.set_ylabel('Actual',color='#8b92a5')
        ax.tick_params(colors='#8b92a5')
        plt.title('Confusion Matrix',color='#e0e0ff',pad=12)
        st.pyplot(fig, use_container_width=True)

    with feat_col:
        st.markdown("**Feature Importances**")
        imp_sorted = importances.sort_values(ascending=True)
        colors = ['#7c8cf8' if v>imp_sorted.median() else '#4a5296' for v in imp_sorted]
        fig2 = go.Figure(go.Bar(
            x=imp_sorted.values, y=imp_sorted.index,
            orientation='h', marker_color=colors,
            text=[f"{v:.3f}" for v in imp_sorted.values], textposition='outside',
        ))
        fig2.update_layout(plot_bgcolor='#1a1f33',paper_bgcolor='#1a1f33',
                           font_color='#e0e0e0',height=340,
                           xaxis_title="Importance Score",
                           margin=dict(t=10,b=30,l=10,r=60))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Cross-Validation Scores (5-Fold)**")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(5)], y=cv_scores*100,
                          marker_color='#7c8cf8',
                          text=[f"{s:.1f}%" for s in cv_scores*100], textposition='outside'))
    fig3.add_hline(y=cv_scores.mean()*100, line_dash='dash', line_color='#52b788',
                   annotation_text=f"Mean: {cv_scores.mean():.1%}", annotation_font_color='#52b788')
    fig3.update_layout(plot_bgcolor='#1a1f33',paper_bgcolor='#1a1f33',
                       font_color='#e0e0e0',height=260,yaxis_range=[0,110],
                       yaxis_title="Accuracy (%)",margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)

    race_rev   = {v:k for k,v in RACE_MAP.items()}
    lunch_rev  = {v:k for k,v in LUNCH_MAP.items()}
    att_rev    = {v:k for k,v in ATTENDANCE_MAP.items()}
    course_rev = {v:k for k,v in COURSE_MAP.items()}

    df_d = df.copy()
    df_d['race/ethnicity']              = df_d['race/ethnicity'].map(race_rev)
    df_d['lunch']                       = df_d['lunch'].map(lunch_rev)
    df_d['test preparation course']     = df_d['test preparation course'].map(att_rev)
    df_d['parental level of education'] = df_d['parental level of education'].map(course_rev)

    r1c1,r1c2 = st.columns(2)
    with r1c1:
        st.markdown("**Score Distribution by Gender**")
        fig = px.box(df_d,x='gender',y='average score',color='gender',
                     color_discrete_map={'male':'#7c8cf8','female':'#e76f51'},
                     template='plotly_dark')
        fig.update_layout(plot_bgcolor='#1a1f33',paper_bgcolor='#1a1f33',
                          height=300,margin=dict(t=10,b=10,l=10,r=10),showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown("**Pass Rate by Attendance**")
        att_pass = df_d.groupby('test preparation course')['result'].mean().reset_index()
        att_pass.columns = ['Attendance','Pass Rate']
        att_pass['Pass Rate'] *= 100
        fig = px.bar(att_pass,x='Attendance',y='Pass Rate',color='Attendance',
                     color_discrete_sequence=['#e76f51','#52b788'],
                     template='plotly_dark',text='Pass Rate')
        fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
        fig.update_layout(plot_bgcolor='#1a1f33',paper_bgcolor='#1a1f33',
                          height=300,margin=dict(t=10,b=10,l=10,r=10),
                          showlegend=False,yaxis_range=[0,110])
        st.plotly_chart(fig, use_container_width=True)

    r2c1,r2c2 = st.columns(2)
    with r2c1:
        st.markdown("**Average Score by Pursuing Course**")
        edu_avg = df_d.groupby('parental level of education')['average score'].mean().sort_values()
        fig = px.bar(x=edu_avg.values,y=edu_avg.index,orientation='h',
                     color=edu_avg.values,color_continuous_scale='Blues',
                     template='plotly_dark')
        fig.update_layout(plot_bgcolor='#1a1f33',paper_bgcolor='#1a1f33',
                          height=320,margin=dict(t=10,b=10,l=10,r=10),
                          coloraxis_showscale=False,xaxis_range=[0,100])
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        st.markdown("**Math vs Reading Scores (by Result)**")
        fig = px.scatter(df_d,x='math score',y='reading score',
                         color=df_d['result'].map({1:'Pass',0:'Fail'}),
                         color_discrete_map={'Pass':'#52b788','Fail':'#e76f51'},
                         opacity=0.6,template='plotly_dark')
        fig.update_layout(plot_bgcolor='#1a1f33',paper_bgcolor='#1a1f33',
                          height=320,margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Pass Rate by Community**")
    race_pass = df_d.groupby('race/ethnicity')['result'].mean().reset_index()
    race_pass.columns=['Community','Pass Rate']; race_pass['Pass Rate']*=100
    fig = px.bar(race_pass.sort_values('Pass Rate'),x='Community',y='Pass Rate',
                 color='Pass Rate',color_continuous_scale='Blues',
                 template='plotly_dark',text='Pass Rate')
    fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
    fig.update_layout(plot_bgcolor='#1a1f33',paper_bgcolor='#1a1f33',
                      height=280,margin=dict(t=10,b=10,l=10,r=10),
                      coloraxis_showscale=False,yaxis_range=[0,110])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Pass Rate by Hosteller / Day Scholar**")
    hostel_pass = df_d.groupby('lunch')['result'].mean().reset_index()
    hostel_pass.columns=['Type','Pass Rate']; hostel_pass['Pass Rate']*=100
    fig = px.bar(hostel_pass,x='Type',y='Pass Rate',color='Type',
                 color_discrete_map={'Hosteller':'#7c8cf8','Day Scholar':'#f4a261'},
                 template='plotly_dark',text='Pass Rate')
    fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
    fig.update_layout(plot_bgcolor='#1a1f33',paper_bgcolor='#1a1f33',
                      height=280,margin=dict(t=10,b=10,l=10,r=10),
                      showlegend=False,yaxis_range=[0,110])
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 4 — DATASET
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>📋 Dataset Preview</div>", unsafe_allow_html=True)
    st.markdown(f"""<div class='info-box'>
        📌 <b>Source:</b> Kaggle —
        <a href="https://www.kaggle.com/datasets/spscientist/students-performance-in-exams"
           style="color:#7c8cf8;">Students Performance in Exams</a>
        &nbsp;|&nbsp; <b>Rows:</b> {len(df_raw)}
        &nbsp;|&nbsp; <b>Cols:</b> {len(df_raw.columns)}
        &nbsp;|&nbsp; <b>Target:</b> Pass / Fail (avg ≥ 60)
    </div>""", unsafe_allow_html=True)

    st.dataframe(df_raw.head(50), use_container_width=True, height=400)
    st.markdown("**Statistical Summary**")
    st.dataframe(df_raw.describe().round(2), use_container_width=True)

    csv = df_raw.to_csv(index=False).encode()
    st.download_button("⬇️ Download Dataset as CSV", data=csv,
                       file_name="students_performance.csv", mime="text/csv")
