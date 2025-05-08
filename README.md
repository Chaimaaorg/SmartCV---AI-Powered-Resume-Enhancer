# CV Optimization API

This API enables the extraction, structuring, and AI-enhanced optimization of CVs (resumes) based on job descriptions, with a strong backend foundation and modern technologies such as FastAPI, OCR, and LLMs (Langchain + HuggingFace or Ollama).

---

## 🔧 Overview of the Architecture

| Step                                                                  | Description                                                                                | Resources                                                                                                                                           |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🧠 Backend (FastAPI)                                                  | Solid, fast, and modern Python-based API framework. Type-safe and highly scalable.         | 📚 [FastAPI Documentation](https://fastapi.tiangolo.com/)                                                                                           |
| 📝 Content Extraction (Docling, RapidOCR)                             | Fast OCR suitable for semi-structured documents like resumes.                              | 📚 [Docling GitHub](https://github.com/docling-ai/docling), [RapidOCR GitHub](https://github.com/RapidAI/RapidOCR)                                  |
| 📦 Resume-to-JSON Conversion (LLM via LangChain + HuggingFace/Ollama) | Transforms unstructured CV text into a normalized, structured JSON schema.                 | 📚 [LangChain Docs](https://docs.langchain.dev/), [HuggingFace Hub](https://huggingface.co/models)                                                  |
| 🎯 Resume Optimization (LLM)                                          | Tailors the resume to a specific job description using AI to enhance relevance and impact. | 📚 [McKinsey - AI in Recruiting](https://www.mckinsey.com/capabilities/people-and-organizational-performance/our-insights/the-future-of-recruiting) |
| 🔁 Standardized JSON Output                                           | Enables clean integration with frontend and simplifies rendering and formatting logic.     | —                                                                                                                                                   |
| 🖥️ Frontend (Future Work)                                            | To be defined. Next.js (with App Router) or SvelteKit would be strong modern options.      | 📚 [Next.js Documentation](https://nextjs.org/)                                                                                                     |

---

## 🚀 Why This Project Matters

* ✅ Technically realistic and feasible (popular stack, no exotic dependencies)
* 🎯 Direct impact: improved employability and more relevant job applications
* 🔄 Scalable: future expansion possible (e.g., cover letters, portfolios)
* 🤖 Smart use of LLMs: not rewriting blindly, but targeting the job description intelligently

---

## 💡 Recommendations for Improvement

1. Define a very clear and strict JSON schema from the start using Pydantic. Suggested fields:

   * Personal Info: name, email, phone
   * Experiences
   * Skills
   * Education
   * Certifications
   * Languages

2. Model Selection:

   * Hugging Face: Use Mixtral, Command-R, or T5/T0 for better reformulation
   * Ollama: Convenient for local deployment and privacy-friendly

3. Frontend Strategy:

   * Use Next.js with App Router to build:

     * A page for uploading the CV
     * A job offer selection page
     * A preview of the optimized CV (possibly downloadable in Word or PDF)

4. Bonus Features (Future Scope):

   * Add a scoring system (match score between CV and offer)
   * Generate Word/PDF CVs using python-docx or pdfkit

---

## 📦 Project Features

* 🧾 OCR-based parsing of resumes in PDF format
* 🧠 AI-enhanced resume optimization aligned with job offers
* 📊 Structured data in JSON for easy consumption by frontends
* 📈 Potential for job-CV match scoring and CV generation in various formats

---

## 🔌 Setup Instructions

### Prerequisites

* Python 3.9+
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
* [Ollama](https://ollama.ai/) (optional for LLM processing)

### Installation

```bash
git clone https://github.com/yourusername/cv-optimization-api.git
cd cv-optimization-api

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

To use Ollama with a local model (e.g., Mistral):

```bash
ollama pull mistral
```

---

## 🧪 Usage

### Launch API

```bash
python main.py
```

* Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
* ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### API Endpoints

1. POST /api/process/cv
   Uploads a resume and returns extracted structured data.

```bash
curl -X POST "http://localhost:8000/api/process/cv" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_resume.pdf"
```

2. POST /api/result/optimize
   Sends a structured CV and job offer data, returns an optimized version.

---

## 📁 Project Structure

```
📁 app
├── main.py
├── api/
│   └── v1/
│       └── endpoints/
│           ├── process.py
│           └── result.py
├── models/
│   └── cv_models.py
├── services/
│   ├── cv_parser.py
│   └── cv_optimizer.py
├── temp_files/
└── extracted_markdown/
```

---

## ⚠️ Error Handling

* 400 for invalid file format
* 422 for validation errors
* 500 for internal processing issues (with traceback in dev mode)

---

## 🚀 Deployment Suggestions

* Use Gunicorn with Uvicorn workers:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

* Add rate-limiting, logging, authentication, and env-based config management

---

## 📚 Resources

* [LangChain Structured Output](https://python.langchain.com/docs/modules/model_io/output_parsers/structured)
* [OCR Library Review - TowardsDataScience](https://towardsdatascience.com/top-5-open-source-ocr-libraries-in-2024-9c9a5c04d6d5)
* [RealPython - FastAPI Best Practices](https://realpython.com/fastapi-python-web-apis/)
* [McKinsey - AI in Recruiting](https://www.mckinsey.com/capabilities/people-and-organizational-performance/our-insights/the-future-of-recruiting)
* [Medium - Mastering Structured Output in LLMs](https://medium.com/@docherty/mastering-structured-output-in-llms-choosing-the-right-model-for-json-output-with-langchain-be29fb6f6675)
