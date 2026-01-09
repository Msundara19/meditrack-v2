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
    - Patient-friendly summaries
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
        
        # Build prompt for LLM
        prompt = self._build_prompt(
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
                    area_cm2, redness_index, healing_score, risk_level
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
    
    def _build_prompt(
        self,
        area_cm2: float,
        redness_index: float,
        edge_sharpness: float,
        healing_score: float,
        risk_level: str,
        previous_metrics: Optional[Dict]
    ) -> str:
        """
        Build prompt for LLM analysis
        
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
        prompt = f"""You are a medical AI assistant helping patients monitor their post-surgical wounds. Provide a brief, reassuring, and informative assessment.

**Current Wound Metrics:**
- Wound area: {area_cm2:.2f} cm²
- Redness/inflammation index: {redness_index:.3f} (scale: 0=minimal, 1=high)
- Edge sharpness: {edge_sharpness:.3f} (higher=better defined boundary)
- Overall healing score: {healing_score:.1f}/100
- Risk assessment: {risk_level.upper()}

"""
        
        # Add comparison if previous metrics available
        if previous_metrics:
            prev_area = previous_metrics.get('area_cm2', 0)
            prev_score = previous_metrics.get('healing_score', 0)
            
            area_change = area_cm2 - prev_area
            score_change = healing_score - prev_score
            
            trend = "improving" if score_change > 0 else "stable" if score_change == 0 else "needs attention"
            
            prompt += f"""**Comparison with Previous Scan:**
- Previous area: {prev_area:.2f} cm²
- Area change: {area_change:+.2f} cm² ({("shrinking" if area_change < 0 else "expanding")})
- Healing score change: {score_change:+.1f} points
- Overall trend: {trend}

"""
        
        prompt += """**Please provide:**

1. **Brief Assessment** (2-3 sentences):
   - What do these numbers mean in plain language?
   - Is this normal healing progression?

2. **What to Watch For** (1-2 key points):
   - Specific signs that would indicate a problem
   - When changes are expected

3. **Next Steps** (brief):
   - Should they continue routine care?
   - Do they need to contact their provider?

Keep your response:
- Under 150 words total
- Patient-friendly (no medical jargon)
- Honest but not alarming
- Actionable and specific

Remember: This is for patient education, not diagnosis. Always encourage appropriate medical consultation."""
        
        return prompt
    
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
                    "content": "You are a helpful medical AI assistant providing wound care guidance."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more consistent medical advice
            max_tokens=300,
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
                temperature=0.3,
                max_output_tokens=300,
            )
        )
        
        return response.text.strip()
    
    def _generate_fallback_summary(
        self,
        area_cm2: float,
        redness: float,
        healing_score: float,
        risk_level: str
    ) -> str:
        """
        Generate rule-based summary if LLM APIs fail
        
        Args:
            area_cm2: Wound area
            redness: Redness index
            healing_score: Healing score
            risk_level: Risk level
            
        Returns:
            Generated summary
        """
        summary = f"Your wound measures {area_cm2:.1f} cm² with a healing score of {healing_score:.0f}/100. "
        
        if healing_score >= 70:
            summary += "Your wound appears to be healing well. "
        elif healing_score >= 40:
            summary += "Your wound is showing moderate healing progress. "
        else:
            summary += "Your wound healing may need closer attention. "
        
        if redness > 0.6:
            summary += "There is noticeable redness which could indicate inflammation. "
        
        if risk_level == "high":
            summary += "Based on these metrics, we recommend contacting your healthcare provider for evaluation."
        elif risk_level == "medium":
            summary += "Continue monitoring and contact your provider if you notice any worsening."
        else:
            summary += "Continue your current wound care routine."
        
        return summary
    
    def _generate_recommendations(
        self,
        risk_level: str,
        redness: float,
        area_cm2: float
    ) -> str:
        """
        Generate standardized care recommendations
        
        Args:
            risk_level: Risk assessment level
            redness: Redness index
            area_cm2: Wound area
            
        Returns:
            Recommendation text
        """
        base_care = "Keep the wound clean and dry. Follow your provider's dressing change instructions."
        
        recommendations = [base_care]
        
        # Add specific recommendations based on metrics
        if redness > 0.6:
            recommendations.append(
                "Monitor for signs of infection: increased warmth, swelling, or discharge."
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
                "Contact your provider if symptoms worsen or don't improve in 2-3 days."
            )
        else:
            recommendations.append(
                "Continue monitoring. Contact your provider if you have concerns."
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
