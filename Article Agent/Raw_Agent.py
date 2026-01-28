from langgraph.graph import StateGraph, END
from pydantic import Field, BaseModel
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# llm
load_dotenv()
model_name = "openai/gpt-oss-120b"
llm = ChatGroq(model=model_name)

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
    subject = state["subject"]
    length = state["length"]
    target = state["target"]
    content = state["content"]

    prompt = f"""You are an Article Organizer AI. Your job is to plan and organize an article based on the user's inputs.
    Use the provided state fields:
    - subject: {subject}          # Article topic
    - length: {length}            # how many words in the article
    - target: {target}            # Target audience
    - content: {content}          # Main content or user ideas

    Your tasks:
    1. Fill the `steps` field with 3-5 clear, actionable steps to write this article.
    2. Fill the `instructions_for_writer` field with **detailed instructions** for the Writer Agent.
    - Include article structure: intro, headings, body, conclusion
    - Specify tone and style based on `target`
    - Integrate `subject`, `title`, `header`, `question`, and `content`
    - Make sure the instructions respect the requested `length`

    - `question`: a controversial question in the header so it'd be answered in the main content
    - `steps`: a list of actionable steps
    - `instructions_for_writer`: detailed brief for the writer agent
    """
    print("organizing...⚙️")
    print()
    structured_llm = llm.with_structured_output(State)
    results = structured_llm.invoke(prompt)
    
    return {"Plan":results}

def ArticleWriter(state: dict) -> dict:
    plan = state.get("Plan", "Organizer Agent crushed")
    prompt = f"""
    You're a professionel Article Writer.
    Your role is to help the user write a professionel article based on some given instructions.
    You'll receive those instructions from another agent, and You have to follow the instructions to write the article
    Plan :
    {plan}
    """
    print("Writing the Article...⚙️")
    print()
    result = llm.invoke(prompt)
    
    return {"Article": result.content}

def Structured(state: dict) -> dict:
    prompt = """
    Agent Role: Content Organizer
    Task:
    Receive a text article or blog post draft and restructure it into a well-organized, publication-ready format.
    Instructions:

    Structure the content using clear, hierarchical headings (e.g., H1, H2, H3).
    Use bullet points for lists and key takeaways.
    Remove all word-count notes (e.g., “≈135 words”) and meta-commentary about length.
    Keep all original meaning, data, examples, and emphasis (bold/italics).
    Ensure the flow is logical: Introduction → Main Sections → Conclusion.
    Do not add or remove content—only organize and format.
    Output in clean Markdown.

    Example Input (Short Excerpt):
    “Introduction (≈120 words)
    AI is changing how we teach. It helps personalize lessons. For example, tools like Khan Academy adjust problems based on student level.”

    Example Output:

    Introduction
    AI is changing how we teach. It helps personalize lessons. For example, tools like Khan Academy adjust problems based on student level.
    """

    article = state.get("Article", "")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": article}
    ]
    print("Final Touches...🪶\n")
    print("Article : \n")
    FinalReslt = llm.invoke(messages)
    print(FinalReslt.content)
    return {"Result":FinalReslt.content}

def Reviewer(state: dict) -> dict:
    article = state.get("Result", "")
    instructions = state.get("instructions_for_writer", "")
    prompt = f"""
    You are a professional article reviewer.

    Evaluate the article against the provided instructions.
    Rate it from 0 to 5 (example: 4.6/5).

    Optionally give a simple short note about the rating, just in one line or less.

    Output format:
    Rating: X/5
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"INSTRUCTIONS:\n{instructions}\n\nARTICLE:\n{article}"}
    ]
    # messages = [
    #     {"role": "system", "content": prompt},
    #     {"role": "user", "content": article}
    # ]
    rating = llm.invoke(messages).content
    return {"rating" : rating}


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

Agent = Graph.compile()

state_input = {
    "subject": "subject",
    "length": "length",
    "target": "target",
    "content": "content"
}

