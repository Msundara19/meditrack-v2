"""
LLM Service for AI-Powered Wound Analysis
Integrates Groq (Llama 3.1) and Google Gemini for patient-friendly summaries
"""
import os
import logging
from typing import Optional, Dict
from groq import Groq
import google.generativeai as genai

from app.config import settings

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """
    LLM-powered wound analysis service
    
    Uses:
    - Groq (Llama 3.1-8B) as primary provider for fast inference
    - Google Gemini as fallback provider
    
    Generates:
    - Patient-friendly summaries (in everyday language)
    - Risk assessment
    - Care recommendations
    """
    
    def __init__(self):
        """Initialize LLM clients"""
        # Initialize Groq
        try:
            self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
            logger.info("✓ Groq client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Groq: {e}")
            self.groq_client = None
        
        # Initialize Gemini (optional)
        try:
            if settings.GEMINI_API_KEY:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                logger.info("✓ Gemini client initialized")
            else:
                self.gemini_model = None
                logger.info("Gemini API key not provided, using Groq only")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")
            self.gemini_model = None
    
    def generate_analysis(
        self,
        area_cm2: float,
        redness_index: float,
        edge_sharpness: float,
        healing_score: float,
        previous_metrics: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive wound analysis
        
        Args:
            area_cm2: Wound area in cm²
            redness_index: Redness/inflammation metric (0-1)
            edge_sharpness: Edge definition quality (0-1)
            healing_score: Composite healing score (0-100)
            previous_metrics: Optional previous scan for comparison
            
        Returns:
            Dictionary with:
            - risk_level: "low", "medium", or "high"
            - summary: Patient-friendly analysis
            - recommendations: Care instructions
        """
        # Assess risk level using rule-based logic
        risk_level = self._assess_risk(area_cm2, redness_index, healing_score)
        
        # Build patient-friendly prompt for LLM
        prompt = self._build_patient_friendly_prompt(
            area_cm2=area_cm2,
            redness_index=redness_index,
            edge_sharpness=edge_sharpness,
            healing_score=healing_score,
            risk_level=risk_level,
            previous_metrics=previous_metrics
        )
        
        # Try Groq first, fallback to Gemini
        try:
            summary = self._call_groq(prompt)
            logger.info("Generated analysis using Groq")
        except Exception as e:
            logger.warning(f"Groq failed: {e}, trying Gemini...")
            try:
                summary = self._call_gemini(prompt)
                logger.info("Generated analysis using Gemini")
            except Exception as e2:
                logger.error(f"Both LLM providers failed: {e2}")
                summary = self._generate_fallback_summary(
                    area_cm2, redness_index, healing_score, risk_level, previous_metrics
                )
        
        # Generate recommendations based on risk level
        recommendations = self._generate_recommendations(risk_level, redness_index, area_cm2)
        
        return {
            "risk_level": risk_level,
            "summary": summary,
            "recommendations": recommendations
        }
    
    def _assess_risk(self, area_cm2: float, redness: float, healing_score: float) -> str:
        """
        Rule-based risk assessment
        
        Risk factors:
        - Large wound area (>25 cm²)
        - High redness/inflammation (>0.7)
        - Low healing score (<30)
        
        Args:
            area_cm2: Wound area
            redness: Redness index
            healing_score: Healing score
            
        Returns:
            "low", "medium", or "high"
        """
        high_risk_flags = 0
        medium_risk_flags = 0
        
        # Check area
        if area_cm2 > 25:
            high_risk_flags += 1
        elif area_cm2 > 10:
            medium_risk_flags += 1
        
        # Check redness/inflammation
        if redness > 0.7:
            high_risk_flags += 1
        elif redness > 0.5:
            medium_risk_flags += 1
        
        # Check healing score
        if healing_score < 30:
            high_risk_flags += 1
        elif healing_score < 60:
            medium_risk_flags += 1
        
        # Determine overall risk
        if high_risk_flags >= 2:
            return "high"
        elif high_risk_flags >= 1 or medium_risk_flags >= 2:
            return "medium"
        else:
            return "low"
    
    def _build_patient_friendly_prompt(
        self,
        area_cm2: float,
        redness_index: float,
        edge_sharpness: float,
        healing_score: float,
        risk_level: str,
        previous_metrics: Optional[Dict]
    ) -> str:
        """
        Build patient-friendly prompt for LLM analysis (NO TECHNICAL JARGON)
        
        Args:
            area_cm2: Wound area
            redness_index: Redness metric
            edge_sharpness: Edge quality
            healing_score: Healing score
            risk_level: Assessed risk level
            previous_metrics: Optional previous scan data
            
        Returns:
            Formatted prompt string
        """
        # Convert technical metrics to everyday language
        size_description = self._describe_size(area_cm2)
        size_comparison = self._size_comparison(area_cm2)
        inflammation_level = self._describe_inflammation(redness_index)
        healing_quality = self._describe_healing(healing_score)
        
        prompt = f"""You are a caring, warm healthcare professional talking to a patient about their healing wound. Use simple, everyday language - NO medical jargon or technical measurements.

**Current Situation (DO NOT mention these numbers directly to the patient):**
- Size: {size_description} (approximately {size_comparison})
- Inflammation: {inflammation_level} redness
- Healing progress: The wound is healing {healing_quality}
- Overall score: {healing_score}/100
- Risk level: {risk_level}

"""
        
        # Add comparison if previous metrics available
        if previous_metrics:
            prev_area = previous_metrics.get('area_cm2', 0)
            prev_score = previous_metrics.get('healing_score', 0)
            
            area_change = area_cm2 - prev_area
            score_change = healing_score - prev_score
            
            if area_change < -2:
                prompt += f"**Good News:** The wound is getting smaller since last time (healing well).\n"
            elif area_change > 2:
                prompt += f"**Note:** The wound has grown slightly since last time. This needs attention.\n"
            
            if score_change > 10:
                prompt += f"**Progress:** Healing is improving compared to last scan.\n"
            elif score_change < -10:
                prompt += f"**Concern:** Healing has slowed down since last time.\n"
        
        prompt += """
**Your Task:** Write a brief, warm message (2-3 sentences maximum) that:

1. Explains how the wound is doing in PLAIN ENGLISH
   - Use phrases like "about the size of..." not "6907.2 cm²"
   - Say "healing well" not "healing score of 17/100"
   - Say "some redness" not "redness index 0.58"

2. Sounds like a caring doctor talking to their patient
   - Warm, reassuring tone
   - Be honest but not scary
   - No medical terminology

3. Gives clear next steps
   - "Keep doing what you're doing" OR
   - "Let's have your doctor take a look"

EXAMPLES OF GOOD RESPONSES:
- "Your wound, which is about the size of your palm, is healing steadily. There's a bit of redness, which is normal at this stage. Keep it clean and dry, and continue with your current care routine."

- "I can see your wound is making good progress - it's gotten smaller since last time! The redness has decreased too, which means the inflammation is going down. You're doing great with your care."

- "Your wound is larger than we'd like to see at this stage, and there's noticeable redness around it. I recommend checking in with your doctor in the next day or two to make sure everything is on track."

AVOID SAYING:
- Technical measurements like "6907.2 cm²" or "healing score 17/100"
- Medical jargon like "inflammation index" or "tissue granulation"
- Scary phrases like "HIGH RISK" (say "needs attention" instead)

Write your warm, patient-friendly assessment now:"""
        
        return prompt
    
    def _describe_size(self, area_cm2: float) -> str:
        """Describe wound size in simple terms"""
        if area_cm2 < 1:
            return "very small"
        elif area_cm2 < 5:
            return "small"
        elif area_cm2 < 15:
            return "medium-sized"
        elif area_cm2 < 50:
            return "moderately large"
        else:
            return "large"
    
    def _size_comparison(self, area_cm2: float) -> str:
        """Compare wound size to everyday objects"""
        if area_cm2 < 1:
            return "smaller than a dime"
        elif area_cm2 < 3:
            return "about the size of a quarter"
        elif area_cm2 < 10:
            return "roughly the size of a credit card"
        elif area_cm2 < 25:
            return "about the size of a Post-it note"
        elif area_cm2 < 50:
            return "about the size of your palm"
        elif area_cm2 < 100:
            return "roughly the size of your hand"
        else:
            return "larger than your hand"
    
    def _describe_inflammation(self, redness_index: float) -> str:
        """Describe inflammation in patient-friendly terms"""
        if redness_index < 0.3:
            return "minimal"
        elif redness_index < 0.5:
            return "mild"
        elif redness_index < 0.7:
            return "moderate"
        else:
            return "significant"
    
    def _describe_healing(self, healing_score: float) -> str:
        """Describe healing progress in simple terms"""
        if healing_score >= 80:
            return "very well"
        elif healing_score >= 60:
            return "well"
        elif healing_score >= 40:
            return "at a moderate pace"
        elif healing_score >= 20:
            return "slowly"
        else:
            return "more slowly than expected"
    
    def _call_groq(self, prompt: str) -> str:
        """
        Call Groq API for completion
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
            
        Raises:
            Exception: If API call fails
        """
        if not self.groq_client:
            raise Exception("Groq client not initialized")
        
        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a warm, caring healthcare professional who explains medical information in simple, everyday language. Never use technical jargon or measurements when talking to patients."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,  # Higher for more natural, conversational tone
            max_tokens=200,   # Shorter responses (2-3 sentences)
            top_p=0.9
        )
        
        return response.choices[0].message.content.strip()
    
    def _call_gemini(self, prompt: str) -> str:
        """
        Call Google Gemini API for completion (fallback)
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
            
        Raises:
            Exception: If API call fails
        """
        if not self.gemini_model:
            raise Exception("Gemini model not initialized")
        
        response = self.gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=200,
            )
        )
        
        return response.text.strip()
    
    def _generate_fallback_summary(
        self,
        area_cm2: float,
        redness: float,
        healing_score: float,
        risk_level: str,
        previous_metrics: Optional[Dict] = None
    ) -> str:
        """
        Generate patient-friendly rule-based summary if LLM APIs fail
        
        Args:
            area_cm2: Wound area
            redness: Redness index
            healing_score: Healing score
            risk_level: Risk level
            previous_metrics: Previous scan data
            
        Returns:
            Generated summary in plain language
        """
        size_desc = self._size_comparison(area_cm2)
        healing_desc = self._describe_healing(healing_score)
        
        summary = f"Your wound, which is {size_desc}, is healing {healing_desc}. "
        
        # Add context from previous scan if available
        if previous_metrics:
            prev_area = previous_metrics.get('area_cm2', area_cm2)
            if prev_area > area_cm2 + 2:
                summary += "Good news - it's getting smaller! "
            elif prev_area < area_cm2 - 2:
                summary += "It's grown a bit since last time, which needs attention. "
        
        # Add inflammation context
        if redness > 0.65:
            summary += "There's noticeable redness around the area. "
        elif redness < 0.4:
            summary += "The redness is minimal, which is a good sign. "
        
        # Add action based on risk
        if risk_level == "high":
            summary += "I recommend having your doctor take a look within the next day or two to make sure everything is on track."
        elif risk_level == "medium":
            summary += "Keep monitoring it closely, and reach out to your doctor if you notice any changes or if it doesn't improve in the next few days."
        else:
            summary += "Keep doing what you're doing with your wound care routine - you're on the right track!"
        
        return summary
    
    def _generate_recommendations(
        self,
        risk_level: str,
        redness: float,
        area_cm2: float
    ) -> str:
        """
        Generate patient-friendly care recommendations
        
        Args:
            risk_level: Risk assessment level
            redness: Redness index
            area_cm2: Wound area
            
        Returns:
            Recommendation text in plain language
        """
        recommendations = []
        
        # Basic care
        recommendations.append("Keep the wound clean and dry.")
        recommendations.append("Follow your provider's dressing change instructions.")
        
        # Add specific recommendations based on metrics
        if redness > 0.6:
            recommendations.append(
                "Watch for signs like increased warmth, swelling, or discharge."
            )
        
        if area_cm2 > 15:
            recommendations.append(
                "Avoid activities that might stress the wound area."
            )
        
        # Add urgency based on risk level
        if risk_level == "high":
            recommendations.append(
                "⚠️ Contact your healthcare provider within 24 hours for evaluation."
            )
        elif risk_level == "medium":
            recommendations.append(
                "Contact your provider if things don't improve in the next 2-3 days."
            )
        else:
            recommendations.append(
                "Continue your current care. Reach out to your provider if you have any concerns."
            )
        
        return " ".join(recommendations)


# Singleton instance for reuse
_llm_analyzer = None


def get_llm_analyzer() -> LLMAnalyzer:
    """
    Get singleton LLM analyzer instance
    
    Returns:
        LLMAnalyzer instance
    """
    global _llm_analyzer
    if _llm_analyzer is None:
        _llm_analyzer = LLMAnalyzer()
    return _llm_analyzer