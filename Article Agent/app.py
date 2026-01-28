import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from pydantic import Field, BaseModel
from langchain_groq import ChatGroq

# Design  
st.set_page_config(page_title="AI Editorial Agent", page_icon="✍️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    .stButton>button { background-color: #238636; color: white; width: 100%; border-radius: 8px; font-weight: bold; height: 3em;}
    .result-container { background-color: #1C2128; border: 1px solid #30363D; padding: 2rem; border-radius: 12px; }
    .rating-card { background: linear-gradient(135deg, #1f6feb 0%, #111 100%); padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;}
    .note-card { background-color: #21262D; border-left: 5px solid #238636; padding: 15px; border-radius: 5px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# THE AGENT LOGIC

@st.cache_resource
def get_graph():
    load_dotenv()
    # Using a reliable model name for Groq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.6)

    class State(BaseModel):
        subject: str = Field(description="The subject")
        length: int = Field(description="Length in chars")
        target: str = Field(description="Target audience")
        title: str = Field(description="Controversial title")
        header: str = Field(description="Header")
        question: str = Field(description="Attractive question")
        content: str = Field(description="User content")
        steps: list[str] = Field(default_factory=list, description="Actionable steps")
        instructions_for_writer: str = Field(default="", description="Instructions")

    def OrganizerAgent(state: dict) -> dict:
        # We force the model to ONLY use the tool
        structured_llm = llm.with_structured_output(State)
        
        prompt = f"""You are a Professional Content Strategist.
        You MUST provide your response by filling the tool/schema provided.
        
        User Inputs:
        - Subject: {state['subject']}
        - Target: {state['target']}
        - Max Length: {state['length']} characters
        - Core Ideas: {state['content']}
        
        Fill every field in the schema. Ensure 'instructions_for_writer' is very detailed.
        """
        try:
            results = structured_llm.invoke(prompt)
            return {"Plan": results}
        except Exception as e:
            # Fallback if tool call fails
            st.error(f"Organizer Error: {e}")
            return {"Plan": "Error in planning phase."}

    def ArticleWriter(state: dict) -> dict:
        plan = state.get("Plan")
        # Ensure we are passing a string to the next prompt
        plan_details = plan.json() if hasattr(plan, 'json') else str(plan)
        
        prompt = f"Write a full article following these specific instructions: {plan_details}"
        result = llm.invoke(prompt)
        return {"Article": result.content}

    def Structured(state: dict) -> dict:
        prompt = "Format this text into clean Markdown with H1, H2, and H3 tags. Keep the tone professional. Remove meta-talk."
        res = llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": state.get("Article", "")}])
        return {"Result": res.content}

    def Reviewer(state: dict) -> dict:
        article = state.get("Result", "")
        prompt = "Review this article. Output exactly in this format: \nRating: X/5\nNote: [Your short critique]"
        res = llm.invoke(prompt + f"\n\nArticle:\n{article}")
        return {"rating": res.content}

    # Graph Setup
    workflow = StateGraph(dict)
    workflow.add_node("Organizer", OrganizerAgent)
    workflow.add_node("Writer", ArticleWriter)
    workflow.add_node("Editor", Structured)
    workflow.add_node("Reviewer", Reviewer)
    
    workflow.set_entry_point("Organizer")
    workflow.add_edge("Organizer", "Writer")
    workflow.add_edge("Writer", "Editor")
    workflow.add_edge("Editor", "Reviewer")
    workflow.add_edge("Reviewer", END)
    
    return workflow.compile()

# UI Interface 

def main():
    agent = get_graph()

    with st.sidebar:
        st.title("🚀 Configuration")
        
        subj_list = ["⚽ Sport", "📈 Economics", "🏛️ Politics", "💊 Health", "🍔 Food", "💻 IT", "❤️ Emotional", "Other"]
        sel_subj = st.selectbox("Topic", subj_list)
        final_subj = st.text_input("Enter Topic") if sel_subj == "Other" else sel_subj
        
        target_list = ["👨‍👩‍👧 Family", "👔 Professional", "📱 Social Media", "🏫 School", "🍻 Casual", "🤓 Enthusiasts"]
        final_target = st.selectbox("Audience", target_list)
        final_len = st.slider("Target Chars", 500, 2000, 1200, 100)
        
        st.markdown("---")
        run_btn = st.button("Generate Article")

    st.title("⚡ AI Editorial Agent")
    content_input = st.text_area("What's the article about? (Your ideas)", height=200, placeholder="Write your core message or facts here...")

    if run_btn:
        if not content_input:
            st.error("Please enter some content first!")
            return

        state_input = {
            "subject": final_subj,
            "length": final_len,
            "target": final_target,
            "content": content_input
        }

        with st.status("🛠️ Processing...", expanded=True) as status:
            final_article = ""
            review_text = ""
            
            for output in agent.stream(state_input):
                for key, val in output.items():
                    status.write(f"Step {key} complete...")
                    if key == "Editor":
                        final_article = val.get("Result")
                    if key == "Reviewer":
                        review_text = val.get("rating")
            
            status.update(label="✨ Finished!", state="complete", expanded=False)

        st.markdown("---")
        
        # Parse Rating
        score = "N/A"
        note = "No critique available."
        if "Rating:" in review_text:
            try:
                score = review_text.split("Rating:")[1].split("\n")[0].strip()
                if "Note:" in review_text:
                    note = review_text.split("Note:")[1].strip()
            except:
                score = "Review complete"

        # Rating & Note UI
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f'<div class="rating-card"><h1>{score}</h1><p>Overall Rating</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="note-card"><b>Editor Note:</b><br>{note}</div>', unsafe_allow_html=True)

        # Article
        st.subheader("📝 Final Draft")
        st.markdown(f'<div class="result-container">{final_article}</div>', unsafe_allow_html=True)
        
        st.download_button("Download Markdown", final_article, file_name="article.md")

if __name__ == "__main__":
    main()