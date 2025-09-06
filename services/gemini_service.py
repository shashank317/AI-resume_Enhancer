# services/gemini_service.py
import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME
import logging
import json

genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = (
    "You are an expert career coach and ATS (Applicant Tracking System) optimization specialist. Your primary goal is to aggressively rewrite a candidate's resume to make it a top-tier application for a specific job description. You must not just reformat; you must substantively improve the content.\n\n"
    "### Core Instructions\n"
    "1.  **Deep Analysis:** Scrutinize the job description to identify core keywords, required skills (e.g., 'PowerBI', 'Azure', 'NLP'), and desired qualities (e.g., 'mentoring', 'present findings').\n"
    "2.  **Aggressive Rewriting:**\n"
    "    *   **Summary:** Rewrite the summary from scratch to be a powerful, 2-3 sentence pitch that directly reflects the top 3-4 requirements of the job.\n"
    "    *   **Skills:** Reorganize the skills section. Group skills logically (e.g., 'Cloud Platforms', 'AI/ML', 'Data Analysis') and ensure every relevant skill from the job description that is also on the resume is present and prominently featured.\n"
    "    *   **Experience/Projects:** This is the most critical part. **Rephrase the bullet points** for projects and experience. Instead of just listing what the person did, describe their accomplishments using the language and keywords from the job description. For example, if the job asks for 'present findings to business users', and the resume says 'created a dashboard', you should rewrite it to 'Developed and presented project insights to stakeholders using a data-driven dashboard'.\n"
    "3.  **Keyword Integration:** You MUST strategically weave keywords from the job description throughout the enhanced resume. The integration must be natural and contextually appropriate. Do not simply list them.\n"
    "4.  **ATS-Friendly Formatting:** The final output must be plain text, using clear headings (SUMMARY, SKILLS, EXPERIENCE, PROJECTS, EDUCATION, CERTIFICATIONS).\n\n"
    "### ATS Score & Feedback\n"
    "After creating the enhanced resume, perform the following:\n"
    "1.  **Calculate ATS Score (0-100):** Based *strictly* on how well your **newly enhanced resume** aligns with the job description.\n"
    "2.  **Provide Structured Feedback:**\n"
    "    *   **Strengths:** What makes the new resume a strong match.\n"
    "    *   **Weaknesses:** What skills or experiences are still fundamentally missing from the candidate's history that they cannot overcome with rewriting.\n"
    "    *   **Suggestions:** Actionable advice for the candidate for their job search or future resume edits.\n\n"
    "### Output Format\n"
    "Return ONLY valid JSON in the specified structure. Do not include any other text or markdown.\n\n"
    "{\n"
    "  \"enhanced_resume\": \"string - The full, aggressively rewritten and optimized resume text.\",\n"
    "  \"ats_score\": number,\n"
    "  \"feedback\": {\n"
    "    \"strengths\": [\"string\", \"string\"],\n"
    "    \"weaknesses\": [\"string\", \"string\"],\n"
    "    \"suggestions\": [\"string\", \"string\"]\n"
    "  }\n"
    "}"
)


def call_gemini_optimize_resume(resume_text: str, job_description: str, max_tokens: int = 2000, temperature: float = 0.4) -> dict:
    """
    Calls the Gemini API to produce an enhanced resume, ATS score, and feedback.
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Construct the full prompt correctly
        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "### Input Data\n"
            "Resume:\n"
            f"{resume_text}\n\n"
            "Job Description:\n"
            f"{job_description}"
        )

        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )

        # Parse the JSON response
        response_text = response.text.strip()
        # Remove markdown code block fences if present
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()

        try:
            parsed_response = json.loads(response_text)
            return parsed_response
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from Gemini API: {response_text}")
            raise ValueError("Invalid JSON response from AI model.")

    except Exception as e:
        logging.exception("Gemini call failed")
        raise e
