# Old version 

import streamlit as st
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from pydantic import Field, BaseModel
from langchain_groq import ChatGroq

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(
    page_title="AI Editorial Agent",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a Modern Look
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* Card Styling */
    .css-1r6slb0, .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Box for Results */
    .result-box {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #414141;
        margin-bottom: 20px;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #00ADB5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #00FFF5;
        color: black;
        box-shadow: 0 4px 15px rgba(0, 255, 245, 0.4);
    }

    /* Metric/Rating Styling */
    div[data-testid="stMetricValue"] {
        color: #00ADB5;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. AGENT SETUP (Cached) ---

@st.cache_resource
def get_agent():
    load_dotenv()
    # Using Llama 3 70b or similar as 'openai/gpt-oss-120b' might vary by provider mapping
    # Ensure this model name is correct for your Groq account
    model_name = "openai/gpt-oss-120b" 
    
    try:
        llm = ChatGroq(model=model_name)
    except Exception as e:
        st.error(f"Error initializing Groq: {e}. Check your API Key.")
        return None

    # --- Defined State & Functions (From your code) ---
    class State(BaseModel):
        subject: str = Field(description="The subject field chosen by the user")
        length: int = Field(description="Characters length of the article")
        target: str = Field(description="Target audience")
        title: str = Field(description="A brilliant controversial title")
        header: str = Field(description="Header of the article")
        question: str = Field(description="A controversial attractive question")
        content: str = Field(description="Main content provided by the user")
        steps: list[str] = Field(default_factory=list, description="Steps to follow")
        instructions_for_writer: str = Field(default="", description="Instructions for the writer agent")

    def OrganizerAgent(state: dict) -> dict:
        # Extract inputs
        subject = state.get("subject")
        length = state.get("length")
        target = state.get("target")
        content = state.get("content")

        prompt = f"""You are an Article Organizer AI. Your job is to plan and organize an article based on the user's inputs.
        Use the provided state fields:
        - subject: {subject}
        - length: {length} words
        - target: {target}
        - content: {content}

        Your tasks:
        1. Fill the `steps` field with 3-5 clear, actionable steps.
        2. Fill the `instructions_for_writer` field with **detailed instructions**.
        """
        structured_llm = llm.with_structured_output(State)
        results = structured_llm.invoke(prompt)
        return {"Plan": results}

    def ArticleWriter(state: dict) -> dict:
        plan = state.get("Plan")
        prompt = f"""
        You're a professional Article Writer.
        You'll receive instructions from another agent. Follow them strictly.
        
        Plan & Instructions:
        {plan}
        """
        result = llm.invoke(prompt)
        return {"Article": result.content}

    def Structured(state: dict) -> dict:
        prompt = """
        Agent Role: Content Organizer
        Task: Restructure into well-organized Markdown (H1, H2, bullet points).
        Do not add/remove content. Output clean Markdown only.
        """
        article = state.get("Article", "")
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": article}
        ]
        FinalReslt = llm.invoke(messages)
        return {"Result": FinalReslt.content}

    def Reviewer(state: dict) -> dict:
        article = state.get("Result", "")
        # Access instructions safely
        plan_data = state.get("Plan")
        instructions = plan_data.instructions_for_writer if plan_data else "General professional standards."

        prompt = f"""
        You are a professional article reviewer.
        Evaluate the article against instructions.
        Output format: "Rating: X/5 | Short comment"
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"INSTRUCTIONS:\n{instructions}\n\nARTICLE:\n{article}"}
        ]
        rating_res = llm.invoke(messages).content
        return {"rating": rating_res}

    # --- Graph Construction ---
    Graph = StateGraph(dict)
    Graph.add_node("Organizer", OrganizerAgent)
    Graph.add_node("CntWriter", ArticleWriter)
    Graph.add_node("Structured", Structured)
    Graph.add_node("rating", Reviewer)
    
    Graph.set_entry_point("Organizer")
    Graph.add_edge("Organizer", "CntWriter")
    Graph.add_edge("CntWriter", "Structured")
    Graph.add_edge("Structured", "rating")
    Graph.add_edge("rating", END)
    
    return Graph.compile()

# --- 3. UI LAYOUT ---

def main():
    agent = get_agent()
    
    # Sidebar for Inputs
    with st.sidebar:
        st.header("⚙️ Agent Settings")
        st.markdown("---")
        
        subject = st.text_input("Topic / Subject", placeholder="e.g. AI in Healthcare")
        target_audience = st.text_input("Target Audience", placeholder="e.g. Doctors & Tech Enthusiasts")
        length = st.slider("Approx Word Length", min_value=300, max_value=2000, value=800, step=100)
        
        st.markdown("### 💡 Core Ideas")
        content_input = st.text_area("Draft Notes / Ideas", height=150, placeholder="Paste your rough notes here...")
        
        generate_btn = st.button("🚀 Launch Agent", type="primary")
        
        st.markdown("---")
        st.markdown("*Powered by LangGraph & Groq*")

    # Main Area
    st.title("⚡ AI Editorial Workbench")
    st.markdown("#### From rough ideas to polished, structured articles in seconds.")

    if generate_btn:
        if not subject or not content_input:
            st.warning("⚠️ Please provide at least a Subject and some Content ideas.")
        else:
            inputs = {
                "subject": subject,
                "length": length,
                "target": target_audience,
                "content": content_input
            }
            
            # Container for the process
            status_container = st.status("🤖 Agent Working...", expanded=True)
            
            final_output = {}
            
            try:
                # Streaming the graph execution
                for output in agent.stream(inputs):
                    for key, value in output.items():
                        if key == "Organizer":
                            status_container.write("✅ **Organizer:** Strategy and Outline created.")
                            with status_container.expander("View Plan Details"):
                                st.json(value["Plan"].dict()) # Convert Pydantic to dict for display
                                
                        elif key == "CntWriter":
                            status_container.write("✅ **Writer:** Draft written.")
                            
                        elif key == "Structured":
                            status_container.write("✅ **Editor:** Formatting and structure applied.")
                            final_output["article"] = value["Result"]
                            
                        elif key == "rating":
                            status_container.write("✅ **Reviewer:** Quality check complete.")
                            final_output["rating"] = value["rating"]
                
                status_container.update(label="✨ Process Complete!", state="complete", expanded=False)
                
                # --- RESULTS DISPLAY ---
                st.markdown("---")
                
                # Rating Banner
                if "rating" in final_output:
                    r_col1, r_col2 = st.columns([1, 4])
                    with r_col1:
                        st.metric("Quality Score", final_output["rating"].split("/")[0].replace("Rating:", "").strip())
                    with r_col2:
                        st.info(f"**Reviewer Note:** {final_output['rating']}")

                # Final Article
                st.subheader("📄 Final Article")
                st.markdown(f'<div class="result-box">{final_output.get("article", "")}</div>', unsafe_allow_html=True)
                
                # Download Button
                st.download_button(
                    label="📥 Download Markdown",
                    data=final_output.get("article", ""),
                    file_name=f"{subject.replace(' ', '_')}_article.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()

# new version

import streamlit as st
import os
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from pydantic import Field, BaseModel
from langchain_groq import ChatGroq

# --- 1. STYLE & CONFIG ---
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

# --- 2. THE AGENT LOGIC ---

@st.cache_resource
def get_graph():
    load_dotenv()
    # Using a reliable model name for Groq
    model = "llama-3.3-70b-versatile"
    llm = ChatGroq(model=model, temperature=0.6)

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

# --- 3. THE UI ---

def main():
    agent = get_graph()

    with st.sidebar:
        st.title("🚀 Configuration")
        
        subj_list = ["⚽ Sport", "📈 Economics", "🏛️ Politics", "💊 Health", "🍔 Food", "💻 IT", "❤️ Emotional", "Other"]
        sel_subj = st.selectbox("Topic", subj_list)
        final_subj = st.text_input("Enter Topic") if sel_subj == "Other" else sel_subj
        
        target_list = ["👨‍👩‍👧 Family", "👔 Professional", "📱 Social Media", "🏫 School", "🍻 Casual", "🤓 Enthusiasts"]
        final_target = st.selectbox("Audience", target_list)
        
        # EXACTLY 500 to 2000 as requested
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

        # UI DISPLAY
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