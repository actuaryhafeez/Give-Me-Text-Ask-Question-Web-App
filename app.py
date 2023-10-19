import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Load the RoBERTa model for question-answering
model_name = 'deepset/roberta-base-squad2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

def main():
    st.title("Give Me Text and Ask a Question Web App")

    # Initialize Streamlit session state
    if 'context' not in st.session_state:
        st.session_state.context = ""
    if 'question' not in st.session_state:
        st.session_state.question = ""

    # Text areas for user input using session state
    context = st.text_area("Context", value=st.session_state.context,  height=250)
    question = st.text_input("Question", value=st.session_state.question)

    # Get Answer button
    if st.button('Get Answer'):
        if context and question:
            result = nlp(question=question, context=context)
            st.write(f"Answer: {result['answer']}")
            
            # Clear the input fields after getting the answer
            st.session_state.context = ""
            st.session_state.question = ""
        else:
            st.write("Please provide both context and question.")

if __name__ == "__main__":
    main()
