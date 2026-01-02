# llm-decoder-param-experiments
scripts to run llm generation with varying decoder parameters so we can analyse
how they affect model output

# Goal
The purpose of this work was to fully grok how `temperature`, `top-p`, and
`top-k` parameters for model decoders can affect **what the model generates**
and **how it expresses that**.

Three sets of parameters were used.  one, a deterministic approach that gave us
the model's raw preferences, was the base case.  The second warmed things up a
bit to introduce a small possibility of variance.  The third lowered the model's
confidence drastically and allowed a very high degree of possible variance.

# Outcome
A lot was learned about how prompt structure can affect what the model does, and
how this can differ quite significantly from what the prompter actually
intended.  The obvious lessons around the three paramteres were also learned -
more on that in `experiments.md`

# Tools/HW used
- [MAX framework](https://docs.modular.com/max/) from Modular, to serve the
  `Gemma3-4b-it` model and run a single generation prompt against it.  MAX gives
  us many useful metrics as well as the model output, and offers all the
  required parameters as arguments to its `generate` command
- python scripts for running the test command and parsing the model output
- bash script to run a bunch of tests
- all of this is running on an ordinary server equipped with an NVidia RTX4060ti

# Note:
Unless you're familiar with MAX already, this won't be much use to you.  It is
only intended as a personal record.  I urge you to [see their
tutorials](https://docs.modular.com/max/get-started) for more!
