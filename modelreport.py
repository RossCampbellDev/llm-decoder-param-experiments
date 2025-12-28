from dataclasses import dataclass
import json
from typing import Optional, Dict
import re


@dataclass()
class ModelReport:
    prompt: Optional[str] = None
    max_batch_size: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None

    # Output
    output: Optional[str] = None
    error: Optional[str] = None

    # Performance metrics
    instructions_per_sec: Optional[float] = None
    prompt_size: Optional[int] = None
    output_size: Optional[int] = None
    output_tokens_per_second: Optional[float] = None
    latency_ms: Optional[float] = None
    first_token_latency_ms: Optional[float] = None
    # timestamp: datetime = field(default_factory=datetime.timezone.utcnow)

    @property
    def total_tokens(self) -> Optional[int]:
        if self.prompt_size is None or self.output_size is None:
            return None
        return self.prompt_size + self.output_size

    @property
    def tokens_per_second(self) -> Optional[float]:
        if not self.latency_ms or not self.total_tokens:
            return None
        return self.total_tokens / (self.latency_ms / 1000)

    def as_dict(self) -> Dict:
        return self.__dict__.copy()
    
    def as_json(self) -> str:
        return json.dumps(self.as_dict())
    
    def parse_stdout(self, stdout: str) -> None:
        patterns = {
            "prompt_size": r"Prompt size:\s*(\d+)",
            "output_size": r"Output size:\s*(\d+)",
            "first_token_latency_ms": r"Time to first token:\s*([\d.]+)\s*ms",
            "latency_ms": r"Total Latency:\s*([\d.]+)\s*ms",
            "output_tokens_per_second": (
                r"Eval throughput.*?:\s*([\d.]+)\s*tokens per second"
            ),
            "instructions_per_sec": r"Total Throughput:\s*([\d.]+)\s*req/s",
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, stdout)
            if match:
                value = float(match.group(1))
                if field in {"prompt_size", "output_size"}:
                    value = int(value)
                setattr(self, field, value)

        # Extract generated text (between generation start and metrics)
        split = re.split(r"\nPrompt size:\s*\d+", stdout, maxsplit=1)
        if len(split) == 2:
            body = split[0]
            self.output = (
                body.split("Beginning text generation", 1)[-1].strip().replace("\n", "").replace("\"", "").replace("'", "")
            )

    def parse_error(self, stderr: str) -> None:
        errtext = ""
        for line in stderr.splitlines():
            if "error:" in line.lower():
                errtext += line.strip().replace("\n", "").replace("\"", "").replace("'", "") + ","
        if errtext != "":
            self.error = errtext.strip()
