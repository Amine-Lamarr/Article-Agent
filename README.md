<h1 align="center">⚡ AI Editorial Workbench</h1>

<p align="center">
  <strong>A high-performance Multi-Agent system powered by LangGraph, Groq, and Streamlit.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-LangGraph-blue?style=for-the-badge" alt="LangGraph">
  <img src="https://img.shields.io/badge/Inference-Groq-orange?style=for-the-badge" alt="Groq">
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge" alt="Streamlit">
</p>

<br />

## 📖 Overview
The **AI Editorial Workbench** is a professional-grade content generation pipeline. Unlike standard LLM prompts, this system utilizes a **stateful multi-agent graph** to decompose the writing process into specialized tasks: planning, drafting, formatting, and objective reviewing.



<br />

## 🤖 The Agentic Workflow
The system orchestrates four distinct agents using **LangGraph**:

<table>
  <tr>
    <td><b>Agent</b></td>
    <td><b>Responsibility</b></td>
  </tr>
  <tr>
    <td>📝 <b>Organizer</b></td>
    <td>Analyzes core ideas to build a strategic plan, title, and detailed brief.</td>
  </tr>
  <tr>
    <td>✍️ <b>Writer</b></td>
    <td>Executes the draft based strictly on the Organizer's structural blueprint.</td>
  </tr>
  <tr>
    <td>🎨 <b>Editor</b></td>
    <td>Restructures content into clean, hierarchical Markdown (H1, H2, H3).</td>
  </tr>
  <tr>
    <td>⭐ <b>Reviewer</b></td>
    <td>Scores the final article (0-5) and provides a critical note for quality assurance.</td>
  </tr>
</table>

<br />

## ✨ Key Features
<ul>
  <li><b>Dynamic UI:</b> Modern dark-mode Streamlit interface with real-time status updates.</li>
  <li><b>Precision Control:</b> Hard-coded constraints for character length (500-2000) and target audience mapping.</li>
  <li><b>Structured Output:</b> Pydantic-validated state transitions to prevent hallucination and data mismatch.</li>
  <li><b>Blazing Fast:</b> Leveraging Groq's LPU technology for near-instant generation.</li>
</ul>

<br />

## 🛠️ Installation & Setup

<h3>1. Clone the Repository</h3>

```bash
git clone [https://github.com/yourusername/ai-editorial-agent.git](https://github.com/yourusername/ai-editorial-agent.git)
cd ai-editorial-agent
