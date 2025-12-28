these notes are about running gemma3-4b with various *decoder params* such as `top-p, top-k, seed, temperature, prompt, max-batch-size`
this is so i can observe the difference in model output based on these settings.

i have written a py script for doing `pixi r max generate...` with these params

---
## outputs
the most important outputs are:
**generation** - obviously the generated tokens.
**the decoding params used** - obvious
**token counts** - input, output.  this shows us 'cost intuition', context pressure, truncation risk.  doesn't need to be super accurate
**termination reasons** - did generation stop because we hit some limit?  EOT?  errors?

later we may want more advanced outputs from within the model such as:
- attention weights
- gradients
- loss values
- hidden states

---
# test 1 - why do eclipses happen?
three sets of params:
1. `top-p=1.0, top-k=1, temperature=0.0`
2. `top-p=0.9, top-k=50, temperature=0.2`
3. `top-p=0.95, top-k=100, temperature=0.9`

#### temperature:  how sharp the model's preferences are
rescales logits before sampling
**low temp:  model strongly prefers top choice**
**high temp:  model is more willing to pick alternatives**
if temperature is *too low* it tends to be rigid and repetitive, and mistakes get locked in early
if it is *too high* it tends to hallucinate/drift or be incoherent

**controls confidence, not intelligence**

#### top-k: how many options the model is allowed to consider
'keep only the `k` most likely tokens'
*small k -> constrained, safe, repetitive*
*large k -> more expressive, more risk*
if top-k is *too small* the model will loop and give dull answers
if top-k is *too large* we will generate `rare-token noise`

**top k can prevent the model from wandering off into junk**

#### top-p: how much probability mass is allowed
it keeps the smallest set of tokens whose total probability is `>= p`
*sharp distribution* = few tokens
*flat distribution* = many tokens 

if it's *too low* then it truncates valid continuations
if it's *too high* then it admits low quality tail tokens

**top-p adapts to uncertainty, top-k does not**

> **top-p** responds to *confidence* in an adaptive way, **top-k** does not!
imagine `top-p=0.9` then we have one token with `0.8` probability.  this might only allow 2 or 3 tokens.  whereas if the highest probability token is more like `0.3` then `top-p` will allow lots of tokens

## results
### min temp, high top-p, low top-k
with min temperature it should produce deterministic sort of results.
high top-p should result in a larger range of allowed alternatives
low top-k minimises the number of tokens the model can choose from
*expecting deterministic results where top-p isn't really relevant*
<span style=color:yellow;>This is greedy decoding!</span>
<span style=color:red;>this setup serves as a baseline for testing against.  it shows the model's raw preferences</span>

### low temp, high top-p, high top-k
the low temp will allow a slight willingness to choose alternative tokens
high top-p grants a wider range of alternatives
large top-k gives us a lot of tokens to choose from
*expecting a little bit of variance from case 1, but not much*

**result:** output was actually identical, which suggests that the temperature wasn't high enough to ever result in choosing an alternative token.  this suggests that there was always a very high degree of confidence.  top-p and top-k were essentially redundant

### high temp, high top-p, high top-k
high temp means high willingness to choose alternative tokens
high top-p means lots of tokens if the model is less confident
high top-k means lots of tokens to choose from
*expecting lots more variance from case 1*

**result:** the meaning of the output was the same, and much of the word selection was consistent with the previous two test cases, however there was indeed some difference in generation this time.  

### analysis
our original prompt is quite specific and has specific answers.  the change in temperature largely resulted in a change in *expression* rather than a change in *meaning/knowledge*.
implication: temperature can result in a wide variety of expressions of a small set of meaning, or a small set of expressions of a wider variety of meanings.

**takeaway:** the decoding parameters had more effect on expression than on what the model knows/thinks.  this is because uncertainty was *low*

# test 2 - write a haiku about a random historical figure
in this test we retain the same decoder parameters in each of the 3 test cases, and simply alter the prompt:
`"write a haiku about a random historical figure"`

the results mirror the previous tests in general, with case 1 and 2 being identical and case 3 being only slightly different.
i was expecting a different historical figure to be selected across multiple runs of the various cases, however this did not happen.

**takeaway:** even though i literally instucted the model to select a *random* figure, this did NOT introduce randomness into the output.  likely, it had a selection of probable historical figures but the probability was such that it still chose Cleopatra every time

# test 3 - haiku test but with a specific set of historical figures to choose from
`"write a haiku about a historical figure by sampling from this list: [william shakespeare, arnold palmer, cleopatra, alexander the great, marie curie]"`
the uncertainty that we are looking for needs to be introduced *structurally in the prompt, not linguistically!*

**results:**
1. "Heres a haiku about William Shakespeare:Words flow, rich and deep,Bards tales of love, loss, and might,Stories live and breathe."
2. (same as 1)
3. "Heres a haiku about William Shakespeare, sampling from the list:Words flow, rich and deep,Bards tales of love, loss, and might,Worlds within the verse."

**takeaway:**
though we gave the model a list to now choose from, it still stayed with 1 selection each time.  given the context of writing a haiku - a form of poetry - it seems likely that the model preferred Shakespeare as he is most related to this subject matter

**bonus test:**
I changed the list of names to simply be Roman generals, to hopefully avoid the context of the person affecting the choice.  I also had to specify to the model to choose `one` historical figure, rather than `a`, since it ended up doing haikus about multiple generals - I am not sure why this sudden change.
the new results:
1. "Julius Caesar:  Empires boldest hand,    Crossing Rubicon, a storm,    Romes fate in his grasp."
2. "Julius Caesar:  A leader bold and bright,Empires rise beneath his hand,Fates cruel, final fall."
3. "Quiet strength he bore,Marcus Aurelius, wise sage,Peace within his soul."

Finally!  some more variation.

Test case 1 produced, for some reason, extra whitespace between phrases.  i think this is due to my script stripping whitespace and newlines and converting to and from json, rather than the model.
Test case 2 continued with Julius Caesar, however it wrote a different haiku.  my interpretation is that it is slightly less 'direct' and poetic, but more flowing.
Test case 2 selected a different person!  Hurrah!  it has very different themes to the previous two, showing that the model *framed its responses based heavily on its pseudo-random selection of a person rather than on prompt guidance.*  this is known as `agent branching`

- it's likely that we got Caesar twice since he is by far the most famous and strongly-associated Roman general in the list.  the `0.2` temperature was not enough to overcome the dominant probability of Caesar
- changing the `a` to a `one` was an important change as the former left room for "one or more," whereas the latter introduced a *cardinality constraint*


<h1>the major lesson from these tests is:  Decoding only expresses diversity once the probability distribution is sufficiently flat.  Prompt structure determines whether that flatness exists.</h1>

## how LLMs fail at inference
- Prompt wording is crucial, as the structure of the prompt has more effect than the human interpretation of what it literally says.  For example, specifying "pick a random..." does not necessarily introduce any more randomness
- Increasing temperature helps to flatten the probability distribution of choices, however if one choice initially had a huge probability advantage then even a temperature as high as `0.9` may still result in no diversity of expression
- top-k is a blunt force approach that simply limits the number of options we have (acting as a guard rail), whereas top-p is adaptive to the confidence level of the model.  when we have a steep distribution, top-p allows very few options to choose from and reduces the possibility of a diverse expression
- increasing each of the decoder parameters can allow the model to be more expressive and varied in its output, but at a cost of increased computation.  I haven't observed it, but I also assume that it can increase error rate.
- we must be careful with wording in our prompt.  to human eyes, saying "choose a" may imply a single choice, but when we are working with probabilities and generation this can be expressed by a model as "choose one or more" which was not our intention
