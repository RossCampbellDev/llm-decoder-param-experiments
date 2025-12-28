# llm-decoder-param-experiments
scripts to run llm generation with varying decoder parameters so we can analyse how they affect model output

# Goal
The purpose of this work was to fully grok how `temperature`, `top-p`, and `top-k` parameters for model decoders can affect **what the model generates** and **how it expresses that**.

Three sets of parameters were used.  one, a deterministic approach that gave us the model's raw preferences, was the base case.  The second warmed things up a bit to introduce a small possibility of variance.  The third lowered the model's confidence drastically and allowed a very high degree of possible variance.

# Outcome
A lot was learned about how prompt structure can affect what the model does, and how this can differ quite significantly from what the prompter actually intended.  The obvious lessons around the three paramteres were also learned - more on that in `experiments.md`
