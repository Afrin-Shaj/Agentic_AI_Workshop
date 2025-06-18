import streamlit as st
import os
from dotenv import load_dotenv
from utils.pdf_processor import PDFProcessor
from utils.content_summarizer import ContentSummarizer
from utils.quiz_generator import QuizGenerator

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="Quiz Generator",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Automatic Quiz Generator")
    st.markdown("Upload your study material (PDF) and generate an interactive quiz!")

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please set your GOOGLE_API_KEY in the .env file")
        st.stop()

    # Initialize processors
    pdf_processor = PDFProcessor()
    summarizer = ContentSummarizer(api_key)
    quiz_gen = QuizGenerator(api_key)

    # Session state setup
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = None
    if 'bullet_points' not in st.session_state:
        st.session_state.bullet_points = None
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False

    # Sidebar Upload
    with st.sidebar:
        st.header("📄 Upload Study Material")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        num_questions = st.slider("Number of Questions", 3, 10, 5)

    # Process file and generate quiz
    if uploaded_file is not None:
        if st.button("🔄 Process PDF and Generate Quiz", type="primary"):
            with st.spinner("Processing PDF..."):
                extracted_text = pdf_processor.extract_text_from_pdf(uploaded_file)
                if not extracted_text or not pdf_processor.validate_extracted_text(extracted_text):
                    st.error("Extraction failed or text too short.")
                    return

            with st.spinner("Summarizing content..."):
                bullet_points = summarizer.create_bullet_points(extracted_text)
                if not bullet_points:
                    st.error("Could not generate bullet points.")
                    return
                st.session_state.bullet_points = bullet_points

            with st.spinner("Generating quiz..."):
                quiz_questions = quiz_gen.generate_mcq_quiz(bullet_points, num_questions)
                if not quiz_questions:
                    st.error("Quiz generation failed.")
                    return
                st.session_state.quiz_questions = quiz_questions
                st.session_state.quiz_submitted = False

            st.success("Quiz generated successfully!")

    # Tabs for Question Bank and Quiz
    if st.session_state.quiz_questions:
        tab1, tab2 = st.tabs(["📑 Question Bank", "📝 Take Quiz"])

        # Question Bank tab
        with tab1:
            st.header("📋 Study Material Summary")
            with st.expander("View Bullet Points", expanded=False):
                for i, point in enumerate(st.session_state.bullet_points or [], 1):
                    st.write(f"{i}. {point}")

            st.header("📑 Question Bank (with Answers)")
            for i, q in enumerate(st.session_state.quiz_questions, 1):
                with st.expander(f"Question {i}", expanded=True):
                    st.write(f"**Q:** {q['question']}")
                    for key, val in q['options'].items():
                        st.write(f"{key}: {val}")
                    correct_key = q['correct_answer']
                    st.markdown(f"✅ **Correct Answer:** {correct_key}: {q['options'][correct_key]}")
                    if 'explanation' in q:
                        st.markdown(f"ℹ️ **Explanation:** {q['explanation']}")

        # Take Quiz tab
        with tab2:
            st.header("📝 Quiz Time!")
            st.markdown("**Instructions:** Select the best answer. You need 80% to pass.")

            if not st.session_state.quiz_submitted:
                with st.form("quiz_form"):
                    user_answers = {}
                    for i, question in enumerate(st.session_state.quiz_questions):
                        st.subheader(f"Question {i+1}")
                        st.write(question['question'])

                        answer = st.radio(
                            "Select your answer:",
                            options=[""] + list(question['options'].keys()),
                            format_func=lambda x: f"{x}: {question['options'][x]}" if x else "Select an option",
                            key=f"q_{i}"
                        )
                        user_answers[f"q_{i}"] = answer

                    submitted = st.form_submit_button("Submit Quiz", type="primary")
                    if submitted:
                        unanswered = [k for k, v in user_answers.items() if v == ""]
                        if unanswered:
                            st.warning("Please answer all questions before submitting.")
                        else:
                            score_info = quiz_gen.calculate_score(user_answers, st.session_state.quiz_questions)
                            st.session_state.quiz_submitted = True

                            # Show Results
                            st.header("📊 Quiz Results")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Score", f"{score_info['correct']}/{score_info['total']}")
                            col2.metric("Percentage", f"{score_info['percentage']:.1f}%")
                            status = "✅ PASSED" if score_info['passed'] else "❌ FAILED"
                            col3.metric("Status", status)

                            # Review Answers
                            st.subheader("📝 Review Answers")
                            for i, question in enumerate(st.session_state.quiz_questions):
                                q_key = f"q_{i}"
                                user_ans = user_answers.get(q_key, "Not answered")
                                correct_ans = question['correct_answer']
                                is_correct = user_ans == correct_ans

                                with st.expander(f"Question {i+1} {'✅' if is_correct else '❌'}", expanded=True):
                                    st.write(f"**Question:** {question['question']}")
                                    st.write(f"**Your Answer:** {user_ans} - {question['options'].get(user_ans, 'Not answered')}")
                                    st.write(f"**Correct Answer:** {correct_ans} - {question['options'][correct_ans]}")
                                    if 'explanation' in question:
                                        st.write(f"**Explanation:** {question['explanation']}")

            # Reset quiz button
            if st.button("🔄 Generate New Quiz"):
                st.session_state.quiz_questions = None
                st.session_state.bullet_points = None
                st.session_state.quiz_submitted = False
                st.experimental_rerun()

if __name__ == "__main__":
    main()
