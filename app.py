import os
import re
import json
import pdfplumber
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
PAPER_DIR = "./pocs_eid_papers/"
DB_DIR = "./chroma_db_eid/"
COUNTRIES = ["Malawi", "Zimbabwe", "South Africa", "Kenya", "Mozambique", 
             "Lesotho", "Namibia", "Uganda", "Tanzania", "Myanmar", 
             "Papua New Guinea", "Zambia"]

# Enhanced PDF processing functions
def extract_country_from_text(pdf_path):
    """Extract country with improved accuracy"""
    try:
        country_counts = {country: 0 for country in COUNTRIES}
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i > 10:  # Only check first 10 pages
                    break
                text = page.extract_text()
                if text:
                    for country in COUNTRIES:
                        country_counts[country] += text.count(country)
        
        max_country = max(country_counts, key=country_counts.get)
        return max_country if country_counts[max_country] > 0 else "Multi-country"
    except:
        return "Unknown"

def extract_content_metadata(pdf_path):
    """Extract both quantitative and qualitative metadata"""
    metadata = {
        "quantitative": {},
        "qualitative": {
            "themes": [],
            "barriers": [],
            "recommendations": []
        }
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
            
            # Quantitative patterns
            quant_patterns = {
                "cost_per_test": r"\$(\d+(?:\.\d+)?)\s*(?:to|\-)\s*\$(\d+(?:\.\d+)?)",
                "tat_reduction": r"TAT reduced from (\d+)\s* to (\d+)\s*",
                "art_initiation": r"ART initiation (?:increased|rose) (?:from )?(\d+%?) to (\d+%?)",
                "sensitivity": r"sensitivity (?:of )?(\d{2,3}(?:\.\d{1,2})?)%"
            }
            
            for metric, pattern in quant_patterns.items():
                matches = re.findall(pattern, full_text, re.IGNORECASE)
                if matches:
                    flat_matches = list(set([item for sublist in matches for item in sublist if item]))
                    metadata["quantitative"][metric] = flat_matches
            
            # Qualitative patterns
            qual_patterns = {
                "themes": r"(theme|pattern|finding)\s*[:]?\s*'?(.*?)'?(?=[.;\n])",
                "barriers": r"(barrier|challenge|limitation)\s*[:]?\s*'?(.*?)'?(?=[.;\n])",
                "recommendations": r"(recommendation|suggestion|implication)\s*[:]?\s*'?(.*?)'?(?=[.;\n])"
            }
            
            for category, pattern in qual_patterns.items():
                matches = re.findall(pattern, full_text, re.IGNORECASE)
                if matches:
                    # Extract the meaningful part (second group)
                    items = [match[1].strip() for match in matches if match[1].strip()]
                    metadata["qualitative"][category] = items[:5]  # Keep top 5
                    
    except Exception as e:
        print(f"Metadata extraction error: {e}")
    
    return metadata

# Enhanced PDF Processing
def load_and_split_pdfs(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            
            # Extract metadata
            paper_id = filename.replace('.pdf', '')
            year = re.search(r'\d{4}', filename).group(0) if re.search(r'\d{4}', filename) else "Unknown"
            country = extract_country_from_text(path)
            metadata = extract_content_metadata(path)
            
            # Load with PyPDF
            loader = PyPDFLoader(path)
            pdf_docs = loader.load()
            
            # Add enhanced metadata
            for doc in pdf_docs:
                doc.metadata.update({
                    "paper_id": paper_id,
                    "year": year,
                    "country": country,
                    "quantitative": json.dumps(metadata["quantitative"]),
                    "qualitative": json.dumps(metadata["qualitative"]),
                    "content_type": "main_text"
                })
            
            # Text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                separators=["\n\n", "Results:", "Discussion:", "Conclusion:"],
                length_function=len
            )
            split_docs = text_splitter.split_documents(pdf_docs)
            documents.extend(split_docs)
    
    print(f"Loaded {len(documents)} document chunks")
    return documents

# Enhanced Vector Store
def initialize_vector_store(documents):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=DB_DIR,
        collection_name="eid_systematic_review",
    )
    vectorstore.persist()
    return vectorstore

# PhD-Level QA System with Dual Modes
def create_phd_qa_system(vectorstore):
    # Create specialized retrievers
    general_retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    quant_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6, "filter": {"quantitative": {"$ne": "{}"}}}
    )
    qual_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6, "filter": {"qualitative": {"$ne": "{}"}}}
    )
    
    # PhD-Level Prompt Template
    PHD_PROMPT = """
    You are a PhD researcher synthesizing evidence for a dissertation on POC EID. 
    Analyze both quantitative and qualitative dimensions:

    **Quantitative Analysis:**
    1. Report exact numbers with context (values, ranges, statistical measures)
    2. Compare platforms (GeneXpert vs m-PIMA) and settings (tertiary vs primary)
    3. Differentiate geographical contexts (East vs Southern Africa)
    4. Highlight statistical relationships with measures of significance

    **Qualitative Analysis:**
    1. Identify recurring themes and contradictions
    2. Analyze barrier-impact relationships
    3. Develop conceptual frameworks from patterns
    4. Synthesize implementation strategies with contextual nuances

    **Response Structure:**
    - Start with conceptual framework overview
    - Address each research sub-question systematically
    - Present integrated quantitative-qualitative insights
    - Propose theory-informed implementation strategies
    - Note methodological limitations and evidence gaps

    **Quantitative Context:**
    {quant_context}

    **Qualitative Context:**
    {qual_context}

    **General Context:**
    {general_context}

    Question: {question}
    """
    
    prompt = PromptTemplate(
        template=PHD_PROMPT,
        input_variables=["quant_context", "qual_context", "general_context", "question"]
    )
    
    # Formatting functions
    def format_quant_docs(docs):
        formatted = []
        for doc in docs:
            quant_data = json.loads(doc.metadata.get("quantitative", "{}"))
            qual_data = json.loads(doc.metadata.get("qualitative", "{}"))
            themes = ", ".join(qual_data.get("themes", [])[:3]) or "None"
            formatted.append(
                f"Document {doc.metadata['paper_id']} [p.{doc.metadata.get('page', '?')}]:\n"
                f"- Quantitative: {quant_data}\n"
                f"- Qualitative Themes: {themes}"
            )
        return "\n\n".join(formatted)
    
    def format_qual_docs(docs):
        formatted = []
        for doc in docs:
            qual_data = json.loads(doc.metadata.get("qualitative", "{}"))
            barriers = ", ".join(qual_data.get("barriers", [])[:3]) or "None"
            recs = ", ".join(qual_data.get("recommendations", [])[:3]) or "None"
            formatted.append(
                f"Document {doc.metadata['paper_id']} [p.{doc.metadata.get('page', '?')}]:\n"
                f"- Key Barriers: {barriers}\n"
                f"- Recommendations: {recs}"
            )
        return "\n\n".join(formatted)
    
    def format_general_docs(docs):
        return "\n\n".join([f"Document {d.metadata['paper_id']} [p.{d.metadata.get('page', '?')}]: {d.page_content[:300]}..." for d in docs])
    
    # Full chain
    chain = (
        {
            "question": RunnablePassthrough(),
            "quant_context": quant_retriever | format_quant_docs,
            "qual_context": qual_retriever | format_qual_docs,
            "general_context": general_retriever | format_general_docs
        }
        | prompt
        | ChatOpenAI(model="gpt-4-turbo", temperature=0)
        | StrOutputParser()
    )
    
    return chain

# Enhanced Response Formatter
def format_response(response, source_docs):
    result = "## PHD-LEVEL SYNTHESIS\n" + response + "\n\n"
    
    if source_docs:
        result += "## EVIDENCE SOURCES\n"
        source_details = {}
        
        for doc in source_docs:
            source_id = f"{doc.metadata['paper_id']} [p.{doc.metadata.get('page', '?')}]"
            if source_id not in source_details:
                quant_data = json.loads(doc.metadata.get("quantitative", "{}"))
                qual_data = json.loads(doc.metadata.get("qualitative", "{}"))
                
                quant_str = ""
                for metric, values in quant_data.items():
                    if values:
                        quant_str += f"- {metric}: {', '.join(values[:3])}\n"
                
                qual_str = ""
                for category, items in qual_data.items():
                    if items:
                        qual_str += f"- {category}: {', '.join(items[:3])}\n"
                
                source_details[source_id] = {
                    "country": doc.metadata.get("country", "N/A"),
                    "year": doc.metadata.get("year", "N/A"),
                    "quant": quant_str,
                    "qual": qual_str
                }
        
        for i, (source_id, data) in enumerate(source_details.items(), 1):
            result += f"{i}. **{source_id}**\n"
            result += f"   - Country: {data['country']}\n"
            result += f"   - Year: {data['year']}\n"
            
            if data['quant']:
                result += f"   - Quantitative Metrics:\n{data['quant']}"
            if data['qual']:
                result += f"   - Qualitative Insights:\n{data['qual']}"
            
            result += "\n"
    
    return result

# Streamlit Chatbot Interface
def main():
    st.set_page_config(
        page_title="PhD Research Assistant - POC EID",
        page_icon=":microscope:",
        layout="centered"
    )
    
    st.title(":microscope: PhD Research Assistant for POC EID")
    st.caption("Advanced literature synthesis for Early Infant Diagnosis research")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your PhD research assistant for POC EID. Ask me anything about the literature."
        }]
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = None
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Initialize system
    if st.session_state.qa_system is None:
        with st.status("Initializing research assistant...", expanded=True) as status:
            try:
                st.write(":file_folder: Checking vector database...")
                # Check if database exists
                if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
                    st.write(":page_facing_up: Processing PDFs... (This may take several minutes)")
                    papers = load_and_split_pdfs(PAPER_DIR)
                    st.write(":floppy_disk: Creating vector database...")
                    vector_db = initialize_vector_store(papers)
                else:
                    st.write(":mag: Loading existing vector database...")
                    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
                    vector_db = Chroma(
                        persist_directory=DB_DIR,
                        embedding_function=embedding_model,
                        collection_name="eid_systematic_review"
                    )
                
                st.write(":brain: Creating PhD-level QA system...")
                qa_system = create_phd_qa_system(vector_db)
                
                st.session_state.qa_system = qa_system
                st.session_state.vector_db = vector_db
                status.update(label="System ready!", state="complete")
                
            except Exception as e:
                status.update(label=f"Initialization failed: {str(e)}", state="error")
                st.error(f"System initialization failed: {str(e)}")
                return
    
    # User input
    if prompt := st.chat_input("Ask your research question:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Analyzing literature..."):
                try:
                    # Retrieve source documents
                    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 10})
                    source_docs = retriever.get_relevant_documents(prompt)
                    
                    # Get response from QA system
                    response = st.session_state.qa_system.invoke(prompt)
                    
                    # Format the full response with sources
                    full_response = format_response(response, source_docs)
                    
                except Exception as e:
                    full_response = f"ERROR: {str(e)}\n\nPlease try a different question or check your OpenAI API key."
            
            # Display response
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Sidebar controls
    with st.sidebar:
        st.header("Research Parameters")
        st.info(f"**Papers loaded:** {len(os.listdir(PAPER_DIR)) if os.path.exists(PAPER_DIR) else 0}")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat history cleared. Ask a new research question."}
            ]
            st.rerun()
        
        st.divider()
        st.subheader("Instructions")
        st.markdown("""
        1. Ask research questions about POC EID systems
        2. The assistant will synthesize evidence from academic papers
        3. Responses include:
           - PhD-level synthesis
           - Evidence sources with metrics
        4. Example questions:
           - Compare GeneXpert and m-PIMA cost-effectiveness in Malawi
           - What are implementation barriers in primary care settings?
           - How does TAT reduction impact ART initiation?
        """)
        
        st.divider()
        st.caption("System Status: Ready" if st.session_state.qa_system else "Initializing...")

if __name__ == "__main__":
    main()