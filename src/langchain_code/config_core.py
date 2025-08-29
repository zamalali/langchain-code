from __future__ import annotations
import os
import re
import json
import subprocess
from typing import Optional, Dict, Any, Tuple, Dict as _Dict, Any as _Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
def _normalize_gemini_env() -> None:
    """
    Make Gemini work regardless of whether the user set GEMINI_API_KEY or GOOGLE_API_KEY.
    LangChain's ChatGoogleGenerativeAI reads GOOGLE_API_KEY by default, so mirror whichever
    one is present into the other if it's missing.
    """
    gemini = os.environ.get("GEMINI_API_KEY")
    google = os.environ.get("GOOGLE_API_KEY")

    if gemini and not google:
        os.environ["GOOGLE_API_KEY"] = gemini

    if google and not gemini:
        os.environ["GEMINI_API_KEY"] = google

_normalize_gemini_env()


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

        self.openai_models = [
            ModelConfig(
                name="GPT-4o Mini",
                input_cost_per_million=0.15,
                output_cost_per_million=0.60,
                capabilities="Fast + inexpensive general model",
                latency_tier=2,
                reasoning_strength=7,
                context_window=200_000,
                provider="openai",
                model_id="gpt-4o-mini",
                langchain_model_name="gpt-4o-mini",
            ),
            ModelConfig(
                name="GPT-4o",
                input_cost_per_million=5.00,
                output_cost_per_million=15.00,
                capabilities="Higher quality multimodal/chat",
                latency_tier=3,
                reasoning_strength=9,
                context_window=200_000,
                provider="openai",
                model_id="gpt-4o",
                langchain_model_name="gpt-4o",
            ),
        ]
        self.ollama_models = [ 
            ModelConfig( 
                name="Llama 3.1 (Ollama)", 
                input_cost_per_million=0.0, 
                output_cost_per_million=0.0, 
                capabilities="Local default via Ollama", 
                latency_tier=2, 
                reasoning_strength=7, 
                context_window=128_000, 
                provider="ollama", 
                model_id="llama3.1", 
                langchain_model_name="llama3.1", 
            ), 
        ]


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
        elif provider == "openai":
            available = self.openai_models
        elif provider == "ollama":
            available = self.ollama_models
        else:
            raise ValueError(f"Unknown provider: {provider}")

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

def _detect_ollama_models() -> list[str]: 
    """ 
    Probe local Ollama for installed models (best effort, very fast). 
    Returns a list of model names (e.g., ["llama3.1", "mistral", ...]). 
    """ 
    try: 
        p = subprocess.run( 
            ["ollama", "list", "--format", "json"], 
            capture_output=True, text=True, timeout=2 
        ) 
        if p.returncode == 0 and p.stdout.strip(): 
            try: 
                data = json.loads(p.stdout) 
                if isinstance(data, list): 
                    names = [] 
                    for it in data: 
                        # Some versions use "name", some "model" 
                        n = (it.get("name") or it.get("model") or "").strip() 
                        if n: 
                            # Trim tags like ":latest" 
                            names.append(n.split(":")[0]) 
                    return list(dict.fromkeys(names)) 
            except Exception: 
                pass 
        p2 = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=2) 
        if p2.returncode == 0 and p2.stdout: 
            lines = [ln.strip() for ln in p2.stdout.splitlines() if ln.strip()] 
            out = [] 
            for ln in lines[1:]: 
                name = ln.split()[0] 
                if name: 
                    out.append(name.split(":")[0]) 
            return list(dict.fromkeys(out)) 
    except Exception: 
        pass 
    return [] 
 
def _pick_default_ollama_model() -> str:
    env_choice = os.getenv("LANGCODE_OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL")
    if env_choice:
        return env_choice

    names = _detect_ollama_models()
    if "llama3.1" in names:
        return "llama3.1"
    if names:
        return names[0]
    return "llama3.1"


def _cached_chat_model(provider: str, model_name: str, temperature: float = 0.2):
    key = (provider, model_name, temperature)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        m = ChatAnthropic(model=model_name, temperature=temperature)
    elif provider == "gemini":
        gkey = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not gkey:
            raise RuntimeError(
                "Missing GOOGLE_API_KEY / GEMINI_API_KEY. "
                "Set one of these in your .env (same folder you chose as Project)."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        m = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=gkey, transport="rest")
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        m = ChatOpenAI(model=model_name, temperature=temperature)
    elif provider == "ollama": 
        from langchain_ollama import ChatOllama 
        base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") 
        if base_url: 
            m = ChatOllama(model=model_name, temperature=temperature, base_url=base_url) 
        else: 
            m = ChatOllama(model=model_name, temperature=temperature)
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
        if p in {"openai", "gpt"}: 
            return "openai" 
        if p in {"ollama"}: 
            return "ollama"
        return p

    env = os.getenv("LLM_PROVIDER", "gemini").lower()
    if env not in {"gemini", "anthropic", "openai", "ollama"}:
        env = "gemini"
    return env

def get_model(provider: str, query: Optional[str] = None, priority: str = "balanced"):
    """
    When no query is given (no router context), return a solid default:
      - anthropic => 'claude-3-7-sonnet-2025-05-14'
      - gemini    => 'gemini-2.0-flash'
      - openai    => 'gpt-4o-mini'
      - ollama    => detected default (prefers llama3.1)
    With a query, use the speed-first router with reasoning override and cache the model object.
    """
    if not query:
        if provider == "anthropic":
            return _cached_chat_model("anthropic", "claude-3-7-sonnet-2025-05-14", 0.2)
        elif provider == "gemini":
            return _cached_chat_model("gemini", "gemini-2.0-flash", 0.2)
        elif provider == "openai": 
            return _cached_chat_model("openai", "gpt-4o-mini", 0.2) 
        elif provider == "ollama": 
            return _cached_chat_model("ollama", _pick_default_ollama_model(), 0.2)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    optimal = _router.select_optimal_model_for_provider(query, provider, priority)
    if provider == "anthropic":
        return _cached_chat_model("anthropic", optimal.langchain_model_name, 0.2)
    elif provider == "gemini":
        return _cached_chat_model("gemini", optimal.langchain_model_name, 0.2)
    elif provider == "openai": 
        return _cached_chat_model("openai", optimal.langchain_model_name, 0.2) 
    elif provider == "ollama": 
        return _cached_chat_model("ollama", optimal.langchain_model_name, 0.2)
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
        elif provider == "openai": 
            return { 
                'model_name': 'GPT-4o Mini (Default)', 
                'langchain_model_name': 'gpt-4o-mini', 
                'provider': provider, 
                'complexity': 'default', 
                'note': 'Using default model - no query provided for optimization' 
            } 
        elif provider == "ollama": 
            md = _pick_default_ollama_model() 
            return { 
                'model_name': f'{md} (Default)', 
                'langchain_model_name': md, 
                'provider': provider, 
                'complexity': 'default', 
                'note': 'Using locally installed Ollama model' 
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
