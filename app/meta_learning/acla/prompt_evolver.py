"""
Prompt Evolution Engine

Generates improved prompt variations based on performance feedback
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime


class PromptEvolver:
    """
    Evolves prompts using LLM-based meta-reasoning

    Strategies:
    - performance_based: Analyze metrics and adjust approach
    - error_analysis: Focus on specific error patterns
    - ablation: Remove/add components systematically
    - chain_of_thought: Add reasoning scaffolding
    - few_shot_optimization: Improve example selection
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet-20241022"
    ):
        self.llm_client = llm_client
        self.provider = provider
        self.model = model

    async def evolve_prompt(
        self,
        current_prompt: str,
        performance_metrics: Dict[str, float],
        error_examples: List[Dict],
        strategy: str,
        iteration: int
    ) -> str:
        """
        Generate an improved version of the prompt

        Args:
            current_prompt: Current prompt text
            performance_metrics: Performance metrics from evaluation
            error_examples: Examples of errors made
            strategy: Evolution strategy to use
            iteration: Current iteration number

        Returns:
            Improved prompt
        """
        print(f"  Using strategy: {strategy}")

        # Build meta-prompt for prompt improvement
        meta_prompt = self._build_meta_prompt(
            current_prompt=current_prompt,
            metrics=performance_metrics,
            errors=error_examples,
            strategy=strategy,
            iteration=iteration
        )

        # Generate improved prompt using LLM
        if self.llm_client:
            improved_prompt = await self._call_llm(meta_prompt)
        else:
            # Fallback: rule-based evolution
            improved_prompt = self._rule_based_evolution(
                current_prompt, strategy, performance_metrics
            )

        return improved_prompt

    def _build_meta_prompt(
        self,
        current_prompt: str,
        metrics: Dict[str, float],
        errors: List[Dict],
        strategy: str,
        iteration: int
    ) -> str:
        """Build meta-prompt for prompt evolution"""

        base_meta_prompt = f"""You are a prompt engineering expert specializing in iterative prompt optimization.

Your task is to improve the following prompt based on its performance metrics and error analysis.

CURRENT PROMPT (Iteration {iteration}):
{'-'*60}
{current_prompt}
{'-'*60}

PERFORMANCE METRICS:
{json.dumps(metrics, indent=2)}

EVOLUTION STRATEGY: {strategy}

{self._get_strategy_instructions(strategy)}

ERROR ANALYSIS:
{self._format_error_analysis(errors)}

INSTRUCTIONS:
1. Analyze the current prompt's strengths and weaknesses
2. Apply the {strategy} strategy to generate improvements
3. Maintain the core objective while enhancing effectiveness
4. Return ONLY the improved prompt text (no explanations)
5. The improved prompt should be immediately usable

IMPROVED PROMPT:
"""
        return base_meta_prompt

    def _get_strategy_instructions(self, strategy: str) -> str:
        """Get specific instructions for each evolution strategy"""

        strategy_guides = {
            'performance_based': """
STRATEGY: Performance-Based Optimization
- Identify which aspects of the task have low accuracy
- Add explicit instructions for weak areas
- Strengthen task framing and objective clarity
- Consider adding constraints or guidelines
""",
            'error_analysis': """
STRATEGY: Error Analysis
- Focus on patterns in the errors provided
- Add specific guidance to avoid common mistakes
- Include examples of correct handling for error cases
- Refine ambiguous instructions that led to errors
""",
            'ablation': """
STRATEGY: Ablation Testing
- Systematically remove unnecessary complexity
- Identify and eliminate redundant instructions
- Test if adding/removing components improves clarity
- Simplify while maintaining effectiveness
""",
            'chain_of_thought': """
STRATEGY: Chain-of-Thought Enhancement
- Add step-by-step reasoning scaffolding
- Include explicit thought process instructions
- Encourage showing work before answering
- Add verification/validation steps
""",
            'few_shot_optimization': """
STRATEGY: Few-Shot Example Optimization
- If examples exist, improve their quality and diversity
- Add examples if none exist and they would help
- Ensure examples cover edge cases
- Make examples more representative of task distribution
"""
        }

        return strategy_guides.get(strategy, "Apply general optimization principles.")

    def _format_error_analysis(self, errors: List[Dict]) -> str:
        """Format error examples for the meta-prompt"""
        if not errors:
            return "No specific error examples available yet."

        error_text = ""
        for i, error in enumerate(errors[:5], 1):  # Limit to 5 examples
            error_text += f"\nError Example {i}:\n"
            error_text += f"  Input: {error.get('input', 'N/A')}\n"
            error_text += f"  Expected: {error.get('expected', 'N/A')}\n"
            error_text += f"  Got: {error.get('predicted', 'N/A')}\n"
            error_text += f"  Error Type: {error.get('error_type', 'N/A')}\n"

        return error_text

    async def _call_llm(self, meta_prompt: str) -> str:
        """Call LLM API to generate improved prompt"""

        if self.provider == "anthropic":
            response = await self.llm_client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": meta_prompt}
                ]
            )
            return response.content[0].text.strip()

        elif self.provider == "openai":
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a prompt engineering expert."},
                    {"role": "user", "content": meta_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _rule_based_evolution(
        self,
        current_prompt: str,
        strategy: str,
        metrics: Dict[str, float]
    ) -> str:
        """
        Fallback rule-based prompt evolution when no LLM client available
        Simple heuristic improvements
        """

        accuracy = metrics.get('accuracy', 0)

        # Add strategy-specific modifications
        if strategy == 'chain_of_thought':
            if "Let's think step by step" not in current_prompt:
                improved = current_prompt + "\n\nLet's think step by step before answering."
            else:
                improved = current_prompt

        elif strategy == 'performance_based':
            if accuracy < 0.5:
                improved = f"{current_prompt}\n\nBe very careful and double-check your reasoning."
            else:
                improved = current_prompt

        elif strategy == 'ablation':
            # Simple: add conciseness instruction
            improved = f"Be concise and direct.\n\n{current_prompt}"

        else:
            # Default: add emphasis on accuracy
            improved = f"{current_prompt}\n\nFocus on accuracy and precision."

        return improved

    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int = 3
    ) -> List[str]:
        """
        Generate multiple prompt variations for ensemble testing

        Args:
            base_prompt: Base prompt to vary
            num_variations: Number of variations to generate

        Returns:
            List of prompt variations
        """
        variations = [base_prompt]  # Include original

        # Simple variation strategies
        strategies = [
            lambda p: f"{p}\n\nThink carefully before responding.",
            lambda p: f"Task: {p}\n\nProvide your best answer.",
            lambda p: p.replace(".", ".\n"),  # Add line breaks
            lambda p: f"IMPORTANT: {p}",
        ]

        for i in range(min(num_variations - 1, len(strategies))):
            varied = strategies[i](base_prompt)
            variations.append(varied)

        return variations[:num_variations]

    def analyze_prompt_quality(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze prompt quality metrics

        Returns metrics like:
        - Length
        - Clarity score (heuristic)
        - Specificity score
        - Structure score
        """
        words = prompt.split()
        sentences = prompt.split('.')

        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'has_examples': 'example' in prompt.lower(),
            'has_constraints': any(word in prompt.lower() for word in ['must', 'should', 'required']),
            'has_cot': 'step by step' in prompt.lower() or 'reasoning' in prompt.lower(),
            'specificity_score': self._calculate_specificity(prompt)
        }

    def _calculate_specificity(self, prompt: str) -> float:
        """Calculate a heuristic specificity score"""
        specific_words = [
            'specific', 'exactly', 'must', 'required', 'format',
            'example', 'always', 'never', 'only', 'precisely'
        ]

        word_count = len(prompt.split())
        specific_count = sum(1 for word in specific_words if word in prompt.lower())

        return min(specific_count / max(word_count / 100, 1), 1.0)
