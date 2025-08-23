from __future__ import annotations
import os
import re
from typing import Optional, Dict, Any, Tuple, Dict as _Dict, Any as _Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelConfig:
    name: str
    input_cost_per_million: float
    output_cost_per_million: float
    capabilities: str
    latency_tier: int
    reasoning_strength: int
    context_window: int
    provider: str
    model_id: str
    langchain_model_name: str
    context_threshold: Optional[int] = None

class IntelligentLLMRouter:
    """
    Speed-first router with reasoning override:
      - For simple/medium: pick the *fastest* viable model (latency, then cost, then reasoning).
      - For complex/overly_complex: *force strong reasoning* (>=8 / >=9), then pick the *fastest* among them.
    """
    def __init__(self, prefer_lightweight: bool = True):
        self.prefer_lightweight = prefer_lightweight

        # Slightly higher thresholds reduce unnecessary escalation to heavy models.
        if prefer_lightweight:
            self.simple_threshold = 18
            self.medium_threshold = 48
            self.complex_threshold = 78
        else:
            self.simple_threshold = 16
            self.medium_threshold = 42
            self.complex_threshold = 70

        self.complexity_vocabulary = {
            'high_complexity': {
                'microservices', 'architecture', 'distributed', 'blockchain',
                'orchestration', 'kubernetes', 'terraform', 'observability',
                'saga', 'cqrs', 'event-sourcing', 'consensus', 'governance',
                'infrastructure', 'comprehensive', 'enterprise-grade',
                'machine-learning', 'ai', 'neural-network', 'deep-learning'
            },
            'medium_complexity': {
                'implement', 'design', 'optimize', 'integrate', 'refactor',
                'authentication', 'authorization', 'database', 'api', 'framework',
                'pipeline', 'deployment', 'monitoring', 'logging', 'testing',
                'docker', 'container', 'websocket', 'oauth', 'jwt', 'redis',
                'react', 'nodejs', 'typescript', 'migration', 'dashboard'
            },
            'dev_actions': {
                'create', 'build', 'setup', 'configure', 'generate', 'convert',
                'add', 'fix', 'write', 'develop', 'establish'
            }
        }

        self.technical_indicators = {
            'full-stack', 'real-time', 'ci/cd', 'end-to-end', 'e2e', 'unit-test',
            'error-handling', 'load-balancer', 'service-mesh', 'auto-scaling'
        }

        self.question_words = {'how', 'why', 'what', 'when', 'where', 'which', 'who'}
        self.conditional_words = {'if', 'unless', 'provided', 'assuming', 'given', 'suppose'}

        # --- Provider model catalogs ---
        self.gemini_models = [
            ModelConfig(
                name="Gemini 2.0 Flash-Lite",
                input_cost_per_million=0.075,
                output_cost_per_million=0.30,
                capabilities="Smallest, most cost-effective for simple tasks",
                latency_tier=1,
                reasoning_strength=4,
                context_window=1_000_000,
                provider="gemini",
                model_id="gemini-2.0-flash-lite",
                langchain_model_name="gemini-2.0-flash-lite"
            ),
            ModelConfig(
                name="Gemini 2.0 Flash",
                input_cost_per_million=0.10,
                output_cost_per_million=0.40,
                capabilities="Balanced multimodal model for agents",
                latency_tier=1,
                reasoning_strength=6,
                context_window=1_000_000,
                provider="gemini",
                model_id="gemini-2.0-flash",
                langchain_model_name="gemini-2.0-flash"
            ),
            ModelConfig(
                name="Gemini 2.5 Flash-Lite",
                input_cost_per_million=0.10,
                output_cost_per_million=0.40,
                capabilities="Cost-effective with thinking budgets",
                latency_tier=1,
                reasoning_strength=5,
                context_window=1_000_000,
                provider="gemini",
                model_id="gemini-2.5-flash-lite",
                langchain_model_name="gemini-2.5-flash-lite"
            ),
            ModelConfig(
                name="Gemini 2.5 Flash",
                input_cost_per_million=0.30,
                output_cost_per_million=2.50,
                capabilities="Hybrid reasoning model with thinking budgets",
                latency_tier=2,
                reasoning_strength=7,
                context_window=1_000_000,
                provider="gemini",
                model_id="gemini-2.5-flash",
                langchain_model_name="gemini-2.5-flash"
            ),
            ModelConfig(
                name="Gemini 2.5 Pro",
                input_cost_per_million=1.25,
                output_cost_per_million=10.00,
                capabilities="State-of-the-art for coding and complex reasoning",
                latency_tier=3,
                reasoning_strength=10,
                context_window=2_000_000,
                provider="gemini",
                model_id="gemini-2.5-pro",
                langchain_model_name="gemini-2.5-pro",
                context_threshold=200_000
            )
        ]

        self.anthropic_models = [
            ModelConfig(
                name="Claude 3.5 Haiku",
                input_cost_per_million=0.80,
                output_cost_per_million=4.00,
                capabilities="Fastest, most cost-effective model",
                latency_tier=1,
                reasoning_strength=5,
                context_window=200_000,
                provider="anthropic",
                model_id="claude-3-5-haiku-20241022",
                langchain_model_name="claude-3-5-haiku-20241022"
            ),
            ModelConfig(
                name="Claude Sonnet (3.7)",
                input_cost_per_million=3.00,
                output_cost_per_million=15.00,
                capabilities="Optimal balance of intelligence, cost, and speed",
                latency_tier=2,
                reasoning_strength=8,
                context_window=200_000,
                provider="anthropic",
                model_id="claude-3-7-sonnet-20250514",
                langchain_model_name="claude-3-7-sonnet-2025-05-14",
                context_threshold=200_000
            ),
            ModelConfig(
                name="Claude Opus 4.1",
                input_cost_per_million=15.00,
                output_cost_per_million=75.00,
                capabilities="Most intelligent model for complex tasks",
                latency_tier=4,
                reasoning_strength=10,
                context_window=200_000,
                provider="anthropic",
                model_id="claude-opus-4.1-20250501",
                langchain_model_name="claude-opus-4.1-20250501"
            )
        ]

    # ---------- Router internals ----------
    def extract_features(self, query: str) -> Dict[str, Any]:
        if not query:
            return {
                'word_count': 0, 'char_count': 0, 'sentence_count': 0, 'avg_word_length': 0,
                'conjunction_count': 0, 'comma_count': 0, 'nested_clauses': 0, 'question_words': 0,
                'high_complexity_words': 0, 'medium_complexity_words': 0, 'dev_action_words': 0,
                'technical_indicators': 0, 'conditional_words': 0, 'unique_word_ratio': 0,
                'multiple_requests': 0, 'technical_symbols': 0, 'number_count': 0
            }

        ql = query.lower()
        words = re.findall(r'\b\w+\b', ql)
        sentences = re.split(r'[.!?]+', query)

        high_complex_count = sum(1 for w in words if w in self.complexity_vocabulary['high_complexity'])
        medium_complex_count = sum(1 for w in words if w in self.complexity_vocabulary['medium_complexity'])
        dev_action_count = sum(1 for w in words if w in self.complexity_vocabulary['dev_actions'])
        technical_indicator_count = sum(
            1 for term in self.technical_indicators
            if term in ql or term.replace('-', ' ') in ql
        )

        return {
            'word_count': len(words),
            'char_count': len(query),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(w) for w in words) / max(len(words), 1),
            'conjunction_count': len(re.findall(r'\b(and|or|but|however|therefore|moreover|with)\b', ql)),
            'comma_count': query.count(','),
            'nested_clauses': query.count('(') + query.count('['),
            'question_words': sum(1 for w in words if w in self.question_words),
            'high_complexity_words': high_complex_count,
            'medium_complexity_words': medium_complex_count,
            'dev_action_words': dev_action_count,
            'technical_indicators': technical_indicator_count,
            'conditional_words': sum(1 for w in words if w in self.conditional_words),
            'unique_word_ratio': len(set(words)) / max(len(words), 1),
            'multiple_requests': len(re.findall(r'\b(also|additionally|then|next|after|plus|including)\b', ql)),
            'technical_symbols': len(re.findall(r'[{}()[\]<>/\\]', query)),
            'number_count': len(re.findall(r'\d+', query)),
        }

    def calculate_complexity_score(self, features: Dict[str, Any]) -> int:
        score = 0
        wc = features['word_count']
        if wc <= 3:
            score += 2
        elif wc <= 8:
            score += 6
        elif wc <= 15:
            score += 15
        else:
            score += 22 + (wc - 15) * 1.2

        score += features['sentence_count'] * 2.5
        score += features['conjunction_count'] * 2.5
        score += features['comma_count'] * 0.8
        score += features['nested_clauses'] * 4

        score += features['high_complexity_words'] * 7
        score += features['medium_complexity_words'] * 3.5
        score += features['dev_action_words'] * 1.5
        score += features['technical_indicators'] * 4
        score += features['question_words'] * 1.5
        score += features['conditional_words'] * 2.5

        score += features['multiple_requests'] * 3
        score += features['technical_symbols'] * 1.5
        score += min(features['number_count'], 3) * 0.8

        if features['high_complexity_words'] >= 3:
            score += 10
        if features['medium_complexity_words'] >= 4:
            score += 6
        if wc > 20 and features['technical_indicators'] > 0:
            score += 8
        if features['avg_word_length'] > 6:
            score += 4

        return min(int(score), 120)

    def classify_complexity(self, query: str) -> str:
        if not query or not query.strip():
            return "simple"
        features = self.extract_features(query)
        score = self.calculate_complexity_score(features)
        if score <= self.simple_threshold:
            return "simple"
        elif score <= self.medium_threshold:
            return "medium"
        elif score <= self.complex_threshold:
            return "complex"
        return "overly_complex"

    def select_optimal_model_for_provider(self, query: str, provider: str, priority: str = "balanced") -> ModelConfig:
        complexity = self.classify_complexity(query)

        if provider == "gemini":
            available = self.gemini_models
        elif provider == "anthropic":
            available = self.anthropic_models
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # 1) Base candidate pool by complexity (speed-friendly unless complex)
        if complexity == "simple":
            candidates = [m for m in available if m.latency_tier <= 2] or available
        elif complexity == "medium":
            candidates = [m for m in available if m.latency_tier <= 3] or available
        elif complexity in {"complex", "overly_complex"}:
            # Reasoning override: require strong reasoning models
            min_reason = 8 if complexity == "complex" else 9
            candidates = [m for m in available if m.reasoning_strength >= min_reason] or \
                         sorted(available, key=lambda m: (-m.reasoning_strength, m.latency_tier))
        else:
            candidates = available

        # 2) Primary optimization is ALWAYS latency, then cost, then reasoning.
        if priority == "quality" and complexity not in {"complex", "overly_complex"}:
            hiq = [m for m in candidates if m.reasoning_strength >= 8] or candidates
            hiq.sort(key=lambda m: (m.latency_tier, -m.reasoning_strength,
                                    m.input_cost_per_million + m.output_cost_per_million))
            return hiq[0]
        if priority == "cost" and complexity not in {"complex", "overly_complex"}:
            cheap = sorted(candidates, key=lambda m: (m.latency_tier,
                                                      m.input_cost_per_million + m.output_cost_per_million,
                                                      -m.reasoning_strength))
            return cheap[0]

        # default and --speed
        fast = sorted(candidates, key=lambda m: (m.latency_tier,
                                                 m.input_cost_per_million + m.output_cost_per_million,
                                                 -m.reasoning_strength))
        return fast[0]

_router = IntelligentLLMRouter(prefer_lightweight=True)

# ---------- lightweight model cache ----------
_MODEL_CACHE: _Dict[Tuple[str, str, float], _Any] = {}

def _cached_chat_model(provider: str, model_name: str, temperature: float = 0.2):
    key = (provider, model_name, temperature)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        m = ChatAnthropic(model=model_name, temperature=temperature)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        m = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    _MODEL_CACHE[key] = m
    return m

def resolve_provider(cli_llm: str | None) -> str:
    if cli_llm:
        p = cli_llm.lower()
        if p in {"claude", "anthropic"}:
            return "anthropic"
        if p in {"gemini", "google"}:
            return "gemini"
        return p

    env = os.getenv("LLM_PROVIDER", "gemini").lower()
    if env not in {"gemini", "anthropic"}:
        env = "gemini"
    return env

def get_model(provider: str, query: Optional[str] = None, priority: str = "balanced"):
    """
    When no query is given (no router context), return a solid default:
      - anthropic => 'claude-3-7-sonnet-2025-05-14'
      - gemini    => 'gemini-2.0-flash'
    With a query, use the speed-first router with reasoning override and cache the model object.
    """
    if not query:
        if provider == "anthropic":
            return _cached_chat_model("anthropic", "claude-3-7-sonnet-2025-05-14", 0.2)
        elif provider == "gemini":
            return _cached_chat_model("gemini", "gemini-2.0-flash", 0.2)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    optimal = _router.select_optimal_model_for_provider(query, provider, priority)
    if provider == "anthropic":
        return _cached_chat_model("anthropic", optimal.langchain_model_name, 0.2)
    elif provider == "gemini":
        return _cached_chat_model("gemini", optimal.langchain_model_name, 0.2)
    else:
        raise ValueError(f"Unknown provider: {provider}")

def get_model_info(provider: str, query: Optional[str] = None, priority: str = "balanced") -> Dict[str, Any]:
    if not query:
        if provider == "anthropic":
            return {
                'model_name': 'Claude Sonnet (Default)',
                'langchain_model_name': 'claude-3-7-sonnet-2025-05-14',
                'provider': provider,
                'complexity': 'default',
                'note': 'Using default model - no query provided for optimization'
            }
        elif provider == "gemini":
            return {
                'model_name': 'Gemini 2.0 Flash (Default)',
                'langchain_model_name': 'gemini-2.0-flash',
                'provider': provider,
                'complexity': 'default',
                'note': 'Using default model - no query provided for optimization'
            }
        else:
            raise ValueError(f"Unknown provider: {provider}")

    optimal = _router.select_optimal_model_for_provider(query, provider, priority)
    complexity = _router.classify_complexity(query)
    return {
        'model_name': optimal.name,
        'model_id': optimal.model_id,
        'langchain_model_name': optimal.langchain_model_name,
        'provider': provider,
        'complexity': complexity,
        'reasoning_strength': optimal.reasoning_strength,
        'latency_tier': optimal.latency_tier,
        'input_cost_per_million': optimal.input_cost_per_million,
        'output_cost_per_million': optimal.output_cost_per_million,
        'capabilities': optimal.capabilities,
        'context_window': optimal.context_window,
        'priority_used': priority
    }
