
import pandas as pd
import random

random.seed(42)
rows = []

for _ in range(200):
    study_hours = round(random.uniform(0, 10), 1)
    attendance = random.randint(40, 100)
    previous_marks = random.randint(20, 95)

    # Synthetic example data, not real student records
    marks = (
        3.5 * study_hours
        + 0.15 * attendance
        + 0.45 * previous_marks
        + random.uniform(-5, 5)
    )
    marks = round(max(0, min(100, marks)), 1)

    rows.append({
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "final_marks": marks
    })

df = pd.DataFrame(rows)
df.to_csv("student-marks.csv", index=False)

print("Dataset created: student-marks.csv")
print(df.head())
print("Total records:", len(df))
