### Exercise 1 — Fixed N

For the same query and retriever:
- Generate with **Top-1**, **Top-3**, **Top-5** chunks
- Same prompt, same decoding
- Log: correctness, specificity, hallucination You’ll likely see a peak at Top-1
or Top-3.

#### top-5
the first test was with a simplistic prompt that did not enforce the use of our
nicely created *context*.  it showed the model hallucinating and going off-topic
because it was not constrained. it's also likely that the generation would be
hallucinatory as there is a lot of loosely related context rather than specific
info.  this causes *evidence dilution*.  the model fell back to a *strong prior*
which was wrong, but common.

the prompt was modified to enforce the use of the context, with the model being
told to answer "I dont know" if the context does not provide.

**prompt format: **
```
Answer the question using ONLY the context below.
If the answer is not explicitly stated, say "I don't know."

CONTEXT:
<rag_token>

QUESTION:
What restrictions are there?
```
**query:**  `What restrictions are there?`
**rag text:** `The only restrictions are that block-level HTML elements -- e.g. <h3 id="autoescape">Automatic Escaping for Special Characters</h3>  In HTML, there are two characters that demand special treatment: `<` and `&`. In the raw HTML, there's more markup than there is text. 4 &lt; 5  However, inside Markdown code spans and blocks, angle brackets and ampersands are *always* encoded automatically. There is a literal backtick () here.  which will produce this:`
**answer:** `"Block-level HTML elements are the only restrictions."`

**RESULT LOOKS PRETTY GOOD** the answer is somewhat relevant, though mainly just
paraphrasing the salient part of our context.  it does not, however, elaborate.

#### top-3
the top 3 result appears to be more valid and useful:
**rag text:** `The only restrictions are that block-level HTML elements -- e.g. <h3 id=\"autoescape\">Automatic Escaping for Special Characters</h3>  In HTML, there are two characters that demand special treatment: `<` and `&`. In the raw HTML, there's more markup than there is text.`
**answer:** `*   Only block-level HTML elements are allowed.*   The characters < and & must be treated specially.`

**RESULT LOOKS BETTER** having this narrowed focus of context seems to have
helped the model to generate a more complete answer, which seems odd.

#### top-1
top 1 returned a blank output!  this indicates that the model didn't know what
to say, but also failed to meet our described constraint of "answer with 'dont
know' if you do not find an explicit answer".  the model did not understand this
combination of instructions.
