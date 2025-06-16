from flask import Flask, jsonify, render_template, request
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize Flask
app = Flask(__name__)

port = int(os.environ.get("PORT", 5000))
# Set API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Load FAISS vectorstore
DB_FAISS_PATH = "vectordb"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
book_db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Define FAISS retrieval tool
def search_faiss_tool(query: str):
    docs = book_db.similarity_search(query, k=2)
    return "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant book content found."

# Define web search tool
def search_tavily_tool(query: str):
    retriever = TavilySearchAPIRetriever(k=3)
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant web content found."

# Define tools for agent
tools = [
    Tool(
        name="MedicalVectorRetriever",
        func=search_faiss_tool,
        description="Use this to search a medical book for context using patient symptoms and vitals."
    ),
    Tool(
        name="WebSearchRetriever",
        func=search_tavily_tool,
        description="Use this when the book doesn't provide enough context; it performs a web search."
    )
]

# Load LLM
# llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
)

# Initialize Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations = 5,
    handle_parsing_errors=True
)

# Parse agent output
def parse_agent_output(text_data):
    result = {"diseases": [], "tests": [], "tips": []}

    if "POSSIBLE DISEASES:" in text_data and "DIAGNOSTIC TESTS:" in text_data:
        # Split by sections
        diseases_part = text_data.split("DIAGNOSTIC TESTS:")[0].replace("POSSIBLE DISEASES:", "").strip()
        tests_and_tips_part = text_data.split("DIAGNOSTIC TESTS:")[1]
        # print(diseases_part, "first")
        # print(tests_and_tips_part,"second")
        # Check if TIPS section exists
        if "TIPS:" in tests_and_tips_part:
            tests_part = tests_and_tips_part.split("TIPS:")[0].strip()
            tips_part = tests_and_tips_part.split("TIPS:")[1].strip()
        else:
            tests_part = tests_and_tips_part.strip()
            tips_part = ""

        # Parse diseases
        for line in diseases_part.split('\n'):
            if not line.strip().startswith("-"):
                continue
            parts = line.strip().lstrip("- ").split(" - ", 1)
            # print(parts,"parts")
            if len(parts) == 2:
                name, description = parts
                result["diseases"].append({
                    "name": name.strip(),
                    "description": description.strip()
                })
        # print(result, "first")
        # Parse tests (expect 3 parts now: name - description - tips)
        for line in tests_part.split('\n'):
            if not line.strip().startswith("-"):
                continue
            parts = line.strip().lstrip("- ").split(" - ")
            # print(parts,"test-parts")
            if len(parts) == 3:
                name, description, tips = parts
            elif len(parts) == 2:
                name, description = parts
                tips = ""
            else:
                # fallback for unexpected format
                name = parts[0]
                description = "No description"
                tips = ""

            result["tests"].append({
                "name": name.strip(),
                "description": description.strip(),
                "tips": tips.strip()
            })
        # print(result, "second")
        # Parse overall tips section
        # if tips_part:
        #     for line in tips_part.split('\n'):
        #         if not line.strip().startswith("-"):
        #             continue
        #         tip = line.strip().lstrip("- ").strip()
        #         if tip:
        #             result["tips"].append(tip)

    else:
        result["tests"].append({
            "name": "No diagnostic output",
            "description": "Agent output did not match expected format.",
            "tips": ""
        })
    print(result, "hggfffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
    return result


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def recommend():
    try:
        symptoms = request.form.get("symptoms", "")
        vitals = request.form.get("vitals", "")
        age = request.form.get("age", "")
        query = f"Patient Age: {age}. Symptoms: {symptoms}. Vitals: {vitals}"
        print(query, "qury coming")
        prompt = f"""
You are a medical assistant AI that provides:

1. Possible diseases with one-line descriptions based on the patient's age, symptoms, and vitals.
2. Diagnostic test recommendations with one-line justifications considering the patient's specific profile, and additional suitability tips.
3. Important tips or warnings related to the patient's age, health conditions, pregnancy status, or other relevant factors that may affect the suitability or safety of the recommended tests.

Rules:
- If the input does NOT contain valid medical symptoms, vitals, or age, respond with:
  "I cannot provide diagnostic test recommendations or disease names without valid medical symptoms, vitals, and age."
- For valid inputs, follow this output format exactly:

POSSIBLE DISEASES:
- Disease Name - One line description.
- Disease Name - One line description.

DIAGNOSTIC TESTS:
- Test Name - Short reason - Suitability tips or contraindications(eg. not suitable childrerns or pregnant women etc.).

Input query:
Patient Age: {age}. Symptoms: {symptoms}. Vitals: {vitals}
"""

        agent_output = agent.run(prompt)
        parsed_output = parse_agent_output(agent_output)
        # print(agent_output)
        print(parsed_output, "parsed output")
        if not parsed_output.get("tests", []) and not parsed_output.get("diseases", []):
            return jsonify({
                "diseases": [],
                "tests": [{
                    "name": "No diagnostic output",
                    "description": "Agent could not generate any test recommendations or disease names.",
                    "tips": ""
                }]
            })

        return jsonify(parsed_output)

        # if not parsed_output["tests"] and not parsed_output["diseases"]:
        #     return jsonify({
        #         "diseases": [],
        #         "tests": [{
        #             "name": "No diagnostic output",
        #             "description": "Agent could not generate any test recommendations or disease names.",
        #             "tips": ""
        #         }]
        #     })

        # return jsonify(parsed_output)

    except Exception as e:
        print("Agent Error:", str(e))
        return jsonify({
            "diseases": [],
            "tests": [{
                "name": "Internal Error",
                "description": f"An error occurred: {str(e)}",
                "tips": ""
            }]
        })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False,host="0.0.0.0",port=port)
