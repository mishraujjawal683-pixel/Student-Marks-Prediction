
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="📚",
    layout="centered"
)

@st.cache_data
def load_data():
    return pd.read_csv("student-marks.csv")

@st.cache_resource
def train_model():
    df = load_data()

    features = [
        "study_hours",
        "attendance",
        "previous_marks"
    ]
    target = "final_marks"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mae, r2, len(df)

st.title("Student Marks Prediction")
st.write(
    "Study hours, attendance aur previous marks ke basis "
    "par estimated marks predict karein."
)

try:
    model, mae, r2, total_rows = train_model()

    st.caption(f"Training dataset: {total_rows} records")

    with st.form("prediction_form"):
        study_hours = st.slider(
            "Daily study hours", 0.0, 12.0, 4.0, 0.5
        )
        attendance = st.slider(
            "Attendance (%)", 0, 100, 75
        )
        previous_marks = st.slider(
            "Previous exam marks", 0, 100, 60
        )

        submitted = st.form_submit_button("Predict Marks")

    if submitted:
        input_data = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks
        }])

        predicted_marks = float(model.predict(input_data)[0])
        predicted_marks = max(0.0, min(100.0, predicted_marks))

        st.subheader("Prediction Result")
        st.metric("Estimated Marks", f"{predicted_marks:.1f} / 100")

        if predicted_marks >= 75:
            st.write("Estimated performance: Good")
        elif predicted_marks >= 50:
            st.write("Estimated performance: Average")
        else:
            st.write("Estimated performance: Needs improvement")

        st.caption(
            "This is a model estimate, not a guaranteed exam result."
        )

    st.divider()
    st.subheader("Model Evaluation")
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error", f"{mae:.2f}")
    col2.metric("R² score", f"{r2:.3f}")

    st.caption(
        "Evaluation is on a held-out test split. "
        "Because the practice dataset is synthetic, "
        "these metrics do not establish real-world accuracy."
    )

    with st.expander("View dataset"):
        st.dataframe(load_data(), use_container_width=True)

except FileNotFoundError:
    st.error(
        "student-marks.csv nahi mili. Pehle "
        "python generate_dataset.py run karein."
    )
except Exception as e:
    st.error(f"Error: {e}")
