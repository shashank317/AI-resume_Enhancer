# services/gemini_service.py
import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME
import logging
import json

# Configure genai at runtime, not import time.
# This allows for environments like Render to inject secrets before the app runs.

SYSTEM_PROMPT = (
    "You are an expert career coach and ATS (Applicant Tracking System) optimization specialist. Your primary goal is to aggressively rewrite a candidate's resume to make it a top-tier application for a specific job description. You must substantively improve the content, not just reformat."
    "\n\n"
    "### Core Instructions\n"
    "1.  **Deep Analysis:** Scrutinize the job description to identify core keywords, required skills (e.g., 'PowerBI', 'Azure', 'NLP'), and desired qualities (e.g., 'mentoring', 'present findings'). Differentiate between essential keywords and general filler."
    "2.  **Aggressive Rewriting:**\n"
    "    *   **Summary:** Rewrite the summary from scratch to be a powerful, 2-3 sentence pitch that directly reflects the top 3-4 requirements of the job.\n"
    "    *   **Experience/Projects:** This is the most critical part. **Rephrase bullet points** to mirror the language and keywords of the job description. Transform duties into accomplishments. For example, if the job asks for 'present findings to business users', and the resume says 'created a dashboard', you must rewrite it to 'Developed and presented project insights to stakeholders using a data-driven dashboard, leading to a 15% improvement in decision-making efficiency'. Use the STAR method (Situation, Task, Action, Result) where possible."
    "3.  **Keyword Integration:** Strategically weave the most important keywords from the job description throughout the enhanced resume. The integration must be natural and contextually appropriate. Do not simply list them."
    "4.  **ATS-Friendly Formatting:** The final resume output must be plain text, using clear headings (e.g., SUMMARY, SKILLS, EXPERIENCE, PROJECTS, EDUCATION)."
    "\n\n"
    "### Analysis and Feedback Generation\n"
    "After creating the enhanced resume, perform the following analysis and return it in the JSON structure:"
    "1.  **ATS Breakdown:** Provide a score from 0-100 for each of the following categories based on how well your **newly enhanced resume** aligns with the job description:\n"
    "    *   `Skills Match`: How well the skills section and skills mentioned in the text match the job's requirements."
    "    *   `Experience Relevance`: How relevant the work/project experience is to the role."
    "    *   `Education Alignment`: How well the education section aligns with the job's requirements."
    "    *   `Keyword Coverage`: The percentage of critical job description keywords present in the new resume."
    "2.  **Keyword Analysis:**\n"
    "    *   `matched_keywords`: A list of the top 5-10 most important keywords from the job description that are now present in the enhanced resume."
    "    *   `missing_keywords`: A list of the top 5-10 most important keywords from the job description that are still missing from the resume (because the candidate lacks the skill/experience)."
    "3.  **Structured Feedback:**\n"
    "    *   `strengths`: 2-3 bullet points on what makes the new resume a strong match."
    "    *   `weaknesses`: 2-3 bullet points on what skills or experiences are still fundamentally missing."
    "    *   `suggestions`: **This is crucial.** Generate 2-3 ready-to-use, impactful resume bullet points for the top missing skills. These should be written in a professional tone, suggesting how the candidate could frame potential experience (e.g., 'Spearheaded the containerization of a monolithic application using Docker, reducing deployment time by 40%')."
    "\n\n"
    "### Output Format\n"
    "You MUST return ONLY a single, valid JSON object. Do not include any other text, explanations, or markdown code fences like ```json. The entire output must be parsable JSON."
    "\n"
    "{\n"
    "  \"enhanced_resume\": \"string - The full, aggressively rewritten and optimized resume text.\",\n"
    "  \"ats_breakdown\": {\n"
    "    \"Skills Match\": number,\n"
    "    \"Experience Relevance\": number,\n"
    "    \"Education Alignment\": number,\n"
    "    \"Keyword Coverage\": number\n"
    "  },\n"
    "  \"matched_keywords\": [\"string\", \"string\"],\n"
    "  \"missing_keywords\": [\"string\", \"string\"],\n"
    "  \"feedback\": {\n"
    "    \"strengths\": [\"string\", \"string\"],\n"
    "    \"weaknesses\": [\"string\", \"string\"],\n"
    "    \"suggestions\": [\"string\", \"string\"]\n"
    "  }\n"
    "}"
)

def call_gemini_optimize_resume(resume_text: str, job_description: str, max_tokens: int = 4096, temperature: float = 0.4) -> dict:
    """
    Calls the Gemini API to produce an enhanced resume and detailed analysis.
    """
    try:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured. Set GEMINI_API_KEY in the environment before calling Gemini APIs.")

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)

        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "### Input Data\n"
            "**Resume to be Optimized:**\n"
            f"{resume_text}\n\n"
            "**Target Job Description:**\n"
            f"{job_description}"
        )

        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                # Ensure the output is JSON
                response_mime_type="application/json",
            )
        )

        # The response should be a valid JSON string now with response_mime_type
        parsed_response = json.loads(response.text)
        return parsed_response

    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from Gemini API response: {response.text}")
        raise ValueError("Invalid JSON response from AI model.") from e
    except Exception as e:
        logging.exception("An unexpected error occurred during the Gemini API call.")
        raise e


def call_gemini_raw(prompt: str, max_tokens: int = 1024, temperature: float = 0.4) -> str:
    """
    Simple wrapper to call Gemini with a free-form prompt and return the raw text output.
    """
    try:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured. Set GEMINI_API_KEY in the environment before calling Gemini APIs.")

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        # model.generate_content returns an object with `.text` containing the generated string
        return getattr(response, 'text', '') or str(response)

    except Exception as e:
        logging.exception("Failed to call Gemini for raw prompt")
        raise e