# FitCoach — Gym Muscle Training Chatbot (Streamlit)
# Save this as app.py (replace your existing file). This version follows the
# same robust import-handling and user-friendly error messages as the SIWS FAQ app.

import sys
import traceback

try:
    import streamlit as st
except Exception:
    print("Missing dependency: streamlit. Run: pip install streamlit")
    raise

st.set_page_config(page_title="FitCoach — Gym FAQ Chatbot", page_icon="🏋️", layout="wide")
st.title("🏋️ FitCoach — Gym Muscle Training Chatbot")
st.markdown("Ask about workouts, exercise form, programming, warmups, nutrition basics, or logging your sessions.")

# Collect missing packages
missing_pkgs = []

# Attempt imports (langchain-community / langchain / transformers / sentence-transformers / faiss / torch)
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document
    from langchain.chains import RetrievalQA
    from langchain_community.llms import HuggingFacePipeline
except Exception:
    missing_pkgs.append("langchain-community")

try:
    import langchain
except Exception:
    missing_pkgs.append("langchain")

try:
    from transformers import pipeline
except Exception:
    missing_pkgs.append("transformers")

try:
    import sentence_transformers  # noqa: F401
except Exception:
    missing_pkgs.append("sentence-transformers")

try:
    import faiss  # noqa: F401
except Exception:
    missing_pkgs.append("faiss-cpu")

try:
    import torch  # noqa: F401
except Exception:
    missing_pkgs.append("torch")

# If packages missing, show helpful message and stop
if missing_pkgs:
    st.error("Some Python packages required by this app are not installed.")
    st.write("Missing packages detected: `" + "`, `".join(missing_pkgs) + "`")
    st.write("To install them, run (in your environment):")
    st.code("pip install " + " ".join(missing_pkgs))
    st.markdown(
        "**Note:** Put package names only in requirements.txt (e.g. `langchain-community`, `transformers`, `sentence-transformers`)."
    )
    st.stop()

# Re-import now that checks passed
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# ---------------- Sample Exercise/Program Data ----------------
EXERCISES = [
    {
        "id": "barbell_back_squat",
        "name": "Barbell Back Squat",
        "muscles": ["Quadriceps", "Glutes", "Hamstrings", "Core"],
        "setup": "Barbell on upper traps, feet shoulder-width, toes slightly out.",
        "cues": [
            "Pack your chest, keep a neutral spine",
            "Knees tracking toes",
            "Sit back into the hips, not down",
            "Drive through heels"
        ],
        "common_mistakes": ["Knees collapsing in", "Rounding the lower back", "Not reaching depth"],
        "recommended": {"sets": "3-5", "reps": "5-8", "rest": "2-3 min"}
    },
    {
        "id": "flat_barbell_bench_press",
        "name": "Barbell Bench Press (Flat)",
        "muscles": ["Pectorals", "Triceps", "Anterior Delts"],
        "setup": "Feet planted, eyes under bar, retract shoulder blades.",
        "cues": ["Tuck elbows slightly on descent", "Drive through feet and press bar up", "Keep scapulae pinched"],
        "common_mistakes": ["Bouncing bar off chest", "Flaring elbows excessively"],
        "recommended": {"sets": "3-5", "reps": "4-8", "rest": "2-3 min"}
    },
    {
        "id": "romanian_deadlift",
        "name": "Romanian Deadlift",
        "muscles": ["Hamstrings", "Glutes", "Lower back"],
        "setup": "Hold bar or dumbbells, slight knee bend, hinge at hips.",
        "cues": ["Push hips back", "Keep bar close to legs", "Maintain long neutral spine"],
        "common_mistakes": ["Rounding back", "Bending knees too much"],
        "recommended": {"sets": "3-4", "reps": "6-10", "rest": "90-120s"}
    },
    {
        "id": "dumbbell_row",
        "name": "One-Arm Dumbbell Row",
        "muscles": ["Lats", "Rhomboids", "Biceps"],
        "setup": "Knee and hand on bench, neutral spine, pull dumbbell to hip.",
        "cues": ["Lead with elbow", "Keep shoulder down", "Brace core"],
        "common_mistakes": ["Using momentum", "Too much trunk rotation"],
        "recommended": {"sets": "3", "reps": "8-12", "rest": "60-90s"}
    }
]

# Convert to langchain Documents
docs = [Document(page_content=ex["name"] + "\n\n" + ex["setup"] + "\n\nCues:\n" + "\n".join(ex["cues"]) + "\n\nCommon mistakes:\n" + "\n".join(ex["common_mistakes"]), metadata={"id": ex["id"], "name": ex["name"]}) for ex in EXERCISES]

# ---------------- Embeddings ----------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

# ---------------- FAISS Vector Store ----------------
try:
    vectorstore = FAISS.from_documents(docs, embeddings)
except Exception:
    st.error("Failed to create FAISS vectorstore. Ensure 'faiss-cpu' and compatible packages are installed.")
    st.write("Detailed error:")
    st.code(traceback.format_exc())
    st.stop()

# ---------------- QA Model (FLAN-T5 Small) ----------------
try:
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
except Exception:
    st.error("Failed to initialize HuggingFace pipeline. Make sure 'transformers' and 'torch' are installed and compatible.")
    st.write("Detailed error:")
    st.code(traceback.format_exc())
    st.stop()

# ---------------- RetrievalQA Chain ----------------
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=True
    )
except Exception:
    st.error("Failed to build RetrievalQA chain. See traceback below.")
    st.code(traceback.format_exc())
    st.stop()

# ---------------- Streamlit UI ----------------
# Sidebar: user profile
st.sidebar.header("Your profile")
age = st.sidebar.number_input("Age", min_value=13, max_value=90, value=28)
sex = st.sidebar.selectbox("Sex", ["male", "female", "other"])
experience = st.sidebar.selectbox("Experience level", ["beginner", "intermediate", "advanced"]) 
goals = st.sidebar.text_input("Primary goal", value="Muscle hypertrophy (gain size)")
days = st.sidebar.slider("Training days per week", 1, 6, 4)
equipment = st.sidebar.multiselect("Equipment available", ["Barbell", "Dumbbells", "Bench", "Squat rack", "Cable machine", "Kettlebell", "Bands", "None"]) 

profile = {"age": age, "sex": sex, "experience": experience, "goals": goals, "days_per_week": days, "equipment": equipment}

# Quick links
st.sidebar.title("📌 Quick Actions")
for item in ["Beginner 3-day split", "Sample warmup", "Squat form tips", "Bench press cues"]:
    st.sidebar.markdown(f"- {item}")

# Conversation state
if "history" not in st.session_state:
    st.session_state.history = []
if "logs" not in st.session_state:
    st.session_state.logs = []

# Main input
query = st.text_input("💬 Ask FitCoach about exercises, programs, nutrition, or logging:")

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Get Advice") and query:
        st.session_state.history.append(("user", query))
        with st.spinner("Thinking like a coach..."):
            try:
                result = qa_chain(query)
            except Exception:
                st.error("Error while running the QA chain. See traceback below.")
                st.code(traceback.format_exc())
                st.stop()

            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            st.session_state.history.append(("assistant", answer))

with col2:
    if st.button("Log sample workout"):
        sample = {"date": __import__("datetime").datetime.now().isoformat(), "notes": "Upper: Bench 3x5 @ RPE8; Rows 3x8", "duration_min": 60}
        st.session_state.logs.append(sample)
        st.success("Sample workout logged.")

# Display conversation
st.subheader("Conversation")
for role, text in st.session_state.history[::-1]:
    if role == "user":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**FitCoach:** {text}")

# Show logs
st.subheader("Workout logs")
if st.session_state.logs:
    import pandas as pd, io
    df = pd.DataFrame(st.session_state.logs)
    st.dataframe(df)
    towrite = io.StringIO()
    df.to_csv(towrite, index=False)
    b = towrite.getvalue().encode()
    st.download_button("Download logs (CSV)", data=b, file_name="workout_logs.csv", mime="text/csv")
else:
    st.info("No workout logs yet — use 'Log sample workout' to add one.")

st.markdown("---")
st.markdown("### Quick exercise reference")
for ex in EXERCISES:
    st.markdown(f"**{ex['name']}** — {', '.join(ex['muscles'])}")
    st.markdown(f"Recommended: {ex['recommended']['sets']} x {ex['recommended']['reps']} (rest {ex['recommended']['rest']})")

# End of app
