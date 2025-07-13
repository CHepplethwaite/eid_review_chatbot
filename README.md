# Systematic Review RAG Assistant

This tool helps you analyze research papers using AI. Simply add your PDFs, ask questions about them, and get instant summaries with sources.

## Requirements
- Computer (Windows/Mac/Linux)
- Internet connection
- OpenAI API key (free trial available)

## Setup Instructions (5 minutes)

### 1. Get your OpenAI API key
- Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Create account if needed
- Click "Create new secret key" and copy it

### 2. Install the software
1. Download this ZIP file and unzip it
2. Open Terminal/Command Prompt:
```bash
pip install -r requirements.txt
```

### 3. Add your API key
Create a new file named `.env` in the main folder with this content:
```env
OPENAI_API_KEY=your_copied_key_here
```

### 4. Add your research papers
- Create a folder named `pocs_eid_papers`
- Place all your PDFs inside this folder

## Using the System

### 1. Run the analysis
In Terminal/Command Prompt:
```bash
python systematic_review.py
```

Wait 2-10 minutes (depending on number of papers) while it processes your documents.

### 2. Ask questions
When prompted, type your research questions like:

```
What do the studies say about turnaround time reduction?
Compare diagnostic accuracy between GeneXpert and m-PIMA
What were the most common operational challenges in African settings?
```

Press Enter to see results

### 3. Understand the output
You'll get two sections:
- **SYSTEMATIC REVIEW FINDINGS**: AI-generated summary
- **EVIDENCE SOURCES**: List of supporting papers with:
  - Paper filename
  - Country of study
  - Excerpt from source text

## Example Questions to Try

1. "What is the average reduction in turnaround time with POC testing?"
2. "Which countries showed the highest ART initiation improvements?"
3. "Compare cost-effectiveness between high-volume and low-volume clinics"
4. "What were the most frequent technical challenges reported?"

## Troubleshooting

- **Missing PDFs?** Ensure they're in `/pocs_eid_papers` folder
- **API key error?** Double-check your `.env` file format
- **Slow processing?** Start with 2-3 papers first
- **Need help?** Contact support@researchai.org
