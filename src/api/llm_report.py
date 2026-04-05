"""
 The sole purpose of this file is to take the raw numerical results from the Vision model
and convert them into a clinical report via the Claude API.
"""

import anthropic
import os


client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

"""
Instead of simply sending abbreviations such as 'MEL' or 'SCC' to a doctor or an AI,
we can establish a better context by sending their full, meaningful names.
"""

CLASS_DESCRIPTIONS = {
    "MEL":  "Melanoma (MEL)",
    "NV":   "Melanocytic Nevus (NV)",
    "BCC":  "Basal Cell Carcinoma (BCC)",
    "AK":   "Actinic Keratosis (AK)",
    "BKL":  "Benign Keratosis-like Lesion (BKL)",
    "DF":   "Dermatofibroma (DF)",
    "VASC": "Vascular Lesion (VASC)",
    "SCC":  "Squamous Cell Carcinoma (SCC)"
}

MALIGNANT_CLASSES = {"MEL", "BCC", "AK" , "SCC"}

def generate_clinical_report(
        diagnosis: str,
        confidence: float,
        all_probs: dict,
        is_risky: bool,
        age: float,
        sex: str,
        anatom_site: str
) -> str:
    """
    The return value is always a string.
    Numerical results are sent to the Claude API and returned as a clinical report.
    Why are we ranking them? If Claude sees the most likely classes first,
    it will formulate the differential diagnosis section more accurately.
    """
    sorted_probs = sorted(
        all_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    prob_lines = []
    for class_code , prob in sorted_probs:
        full_name = CLASS_DESCRIPTIONS.get(class_code, class_code)
        prob_lines.append(f" -{full_name}: {prob * 100:.1f}%")

    prob_table = "\n".join(prob_lines)

    #We are translating the `is_risky` boolean into clinical language that Claude can understand.
    risk_label = "HIGH RISK —  Immediate dermtologist referral recommended" \
                 if is_risky else  \
                 "LOW RISK — Routine monitoring recommended"
    

    prompt = f"""You are an AI-assisted clinical decision support tool specialized in dermoscopic image analysis. Your role is to help dermatologists interpret automated skin lesion classification results. You do NOT replace clinical judgment — you provide structured summaries to support it.

--- PATIENT INFORMATION ---
Age: {int(age)}
Sex: {sex}
Anatomical Site: {anatom_site}

--- MODEL OUTPUT ---
Primary Diagnosis: {CLASS_DESCRIPTIONS.get(diagnosis, diagnosis)}
Confidence: {confidence * 100:.1f}%
Risk Assessment: {risk_label}

Differential Diagnosis (All Class Probabilities):
{prob_table}

--- YOUR TASK ---
Generate a structured clinical report with exactly the following sections:

1. SUMMARY
   One concise paragraph. State the primary diagnosis, confidence level, and risk status.

2. DIFFERENTIAL DIAGNOSIS
   List the top 3 most probable diagnoses based on the probability table above.
   For each: explain why it may or may not be the correct diagnosis given the patient demographics.

3. CLINICAL CONSIDERATIONS
   Based on the anatomical site, patient age, and sex — what clinical factors are relevant?
   Note any demographic risk factors (e.g. age >40 + dorsal site = higher melanoma suspicion).

4. RECOMMENDED NEXT STEPS
   Provide specific actionable steps (dermoscopy, biopsy, follow-up interval, referral urgency).
   Be direct. Avoid vague language like "consider consulting a physician."

5. DISCLAIMER
   One sentence. State that this report is AI-generated and must be reviewed by a licensed dermatologist.

Keep the tone formal, clinical, and precise. Use standard dermatological terminology."""
    
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.content[0].text.strip() #strip() → remove leading and trailing spaces
    
    except anthropic.APIConnectionError:
        # Internet connection or DNS issue
        return "[Clinical report unavailable — API connection error. Model results above are still valid.]"
    
    except anthropic.AuthenticationError:
        # The ANTHROPIC_API_KEY in .env is incorrect or missing
        return "[Clinical report unavailable — Invalid API key. Check ANTHROPIC_API_KEY in .env]"
    
    except anthropic.RateLimitError:
        # Too many requests have been sent — Anthropic has exceeded the limit
        return "[Clinical report unavailable — Rate limit exceeded. Please retry in a moment.]"
    
    except Exception as e:
        return f"[Clinical report unavailable — Unexpected error: {str(e)}]"

