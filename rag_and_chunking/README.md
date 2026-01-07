# Learning the basics of chunking
Chunking is the process of dividing up data/information before embedding it
using a model such as a sentence model. Once we have chunked data, it is
embedded into vectors that collapse down to an individual vector per chunk. We
then compare this vector with our query for similarity, using the result to
select answers to questions.

We have three scripts to help implement some simple chunking:

**chunker.py** contains a `Chunk` class for representing a given chunk of data.
provides functions for different chunking strategies such as fixed size,
structure-aware. fixed size simply chunks by a given number of tokens (overlap
can be used) structure-aware chunking works best on documents such as html,
markdown, yaml, to break into chunks based on the inherent structure of the file
- which should already be quite informative.

**embedder.py** has functions for embedding queries or chunks into vectors,
comparing a query with chunks for a best match, and implementing a basic
similarity. makes use of a simple sentence model to handle the encoding of
chunks.  similarity can also be used, however we have implemented it ourselves.

**tester.py** read a test document (in this case a markdown file full of
information about markdown documents) create a query use the `best_match()`
function from the embedder to select the most ideal chunk for the query.

**semantic_chunker.py** make use of chunking and embedding, and implement a
very simple function to split the test doc into sentences and then embed.
as we go through the doc, we measure the similarity of the next sentence up
with our current "sentence" chunk.  if it's a close match, we include it in
the current chunk.  if not, we start a fresh chunk and carry on

# Flaws
**semantic collision** one of the most obvious flaws with chunking and then
comparing with a query is `semantic collision`. this is where our sentence model
may present a "best match" because it has found similar tokens, but the meaning
in the chunk is not relevant.

we can see an example of this in when querying against our test doc.  if we
query with "please block that quote" - as though looking to have something
removed from a document, our model's best matches talking about block quotes in
markdown formatting.

**chunking strategy weaknesses** using something like structure-aware chunking
seems logical for our markdown-formatted test doc, as the structure of the
document implies something about the meaning of the document - however this can
(and in our case does) result in a large variance in the amount of
data/information in chunks.  sometimes a chunk is one sentence, sometimes it is
multiple paragraphs.  this heavily limits the accuracy of the results - for
example querying "what restrictions are at block level?" should get a near
perfect match in our 'Inline HTML' section, but because the chunk is so big it
seems likely that the model is failing to pick up on that - and is instead
returning much less appropriate responses.

# Comparisons
## fixed size vs structure-aware
**test prompt:** "what restrictions are at block level?" **fixed size 1
params:** size=512, overlap=64 **fixed size 2 params:** size=256, overlap=32

**top-5 for fixed size 1**
0.31674975 `s.    Vestibulum enim wisi, viverra nec,fringilla in, laoreet vitae, risus.    > Donec sit amet nisl. Aliquam semper ipsum sit amet velit. Suspendiss`
0.31063098 `HTML code. (As opposed to raw HTML, which is aterrible format for writing about HTML syntax, because every single < and & in your example code nee`
0.2500959 `om surroundingcontent by blank lines, and the start and end tags of the block shouldnot be indented with tabs or spaces. Markdown is smart enough nott`
0.24820119 `4 spaces or 1 tab. For example, given this input:    This is a normal paragraph:        This is a code block.Markdown will generate:    <p>This is a n`
0.24425378 `ormatting syntax is not processed within block-levelHTML tags. E.g., you can't use Markdown-style *emphasis* inside anHTML block.Span-level HTML tag`

** top-5 for fixed size 2**
0.30437914 `nested (i.e. a blockquote-in-a-blockquote) byadding additional levels of >:    > This is the first level of quoting.    >    > > This is nested blo`
0.29912466 `s. Vestibulum enim wisi, viverra nec, fringilla in, laoreet vitae, risus.    > Donec sit amet nisl. Aliquam semper ipsum sit amet velit. Suspendiss`
0.29290596 `ock Elements](#block)    *   [Paragraphs and Line Breaks](#p)    *[Headers](#header)    *   [Blockquotes](#blockquote)    *   [Lists](#list)    *`
0.29050726 `HTML code. (As opposed to raw HTML, which is aterrible format for writing about HTML syntax, because every single `<`and `&` in your example code nee`
0.27828914 `normal paragraphs, the linesof a code block are interpreted literally. Markdown wraps a code blockin both `<pre>` and `<code>` tags. To produce a code`

**top-5 for structure-aware**
0.36117357 `This is a code block.`
0.30704725 `1986\. What a great season.    <h3 id="precode">Code Blocks</h3>  Pre-formatted code blocks are used for writing about programming or markup sour`
0.30161932 `4<5  However, inside Markdown code spans and blocks, angle brackets and ampersands are *always* encoded automatically. This makes it easy to`
0.2949594 `> This is the first level of quoting.     >     > > This is nested blockquote.>     > Back to the first level.  Blockquotes can contain othe`
0.24914064 `>This is a blockquote         > inside a list item.  To put a code block within a list item, the code block needs`

*it seems fairly obvious that the structure-aware chunks performed far better,
though their similarity scores are not too dissimilar.  However, in this example
we are only taking into account the first 150 characters of each chunk, which
may be poisoning our view.

here are the top 1 scores from each, without trimming them short:*

**fixed size 1:**
```s.    Vestibulum enim wisi, viverra nec, fringilla in,
laoreet vitae, risus.
> Donec sit amet nisl. Aliquam semper ipsum sit amet velit. Suspendisse    id
sem consectetuer libero luctus adipiscing.Blockquotes can be nested (i.e. a
blockquote-in-a-blockquote) byadding additional levels of >:    > This is the
first level of quoting.    >    > > This is nested blockquote.    >    > Back
to the first level.Blockquotes can contain other Markdown elements, including
headers, lists,and code bl
 ```

**fixed size 2:**
```nested (i.e. a blockquote-in-a-blockquote) byadding
additional levels of `>`:    > This is the first level of quoting.    >    > >
This is nested blockquote.    >    > Back to the first l evel.Blockquotes can
contain other Markdown elements, inclu
```

**structure-aware**
`This is a code block.`

### Analysis
something strange is happening.  The query was chosen specifically because it
has a clear match in the doc according to human eyes:  `The only restrictions
are that block-level HTML elements...` and so on. none of the current chunking
strategies bring us to this part of the document. The fixed-size attempts both,
somewhat unsurprisingly, retrieve the same text.  within the 512 and 256 sample
sizes, the `Blockquotes can be nested` portion seems to heavily influence the
resulting vector of the chunk in both cases. There is little similarity in
meaning between the query question and the best match responses of any of these
chunking strategies so far.

### Second attempt
The prompt has been changed to try and engineer a more accurate match:
**prompt:** "what is markdown not a replacement for?"

This *should* find us a chunk with the string "Markdown is not a replacement for
HTML" in it, direct from the test doc.

**fixed size 1 (512, 64):**
0.714856
```rg/doc/To this end, Markdown's syntax is
comprised entirely of punctuationcharacters, which punctuation characters have
been carefully chosen soas to look like what they mean. E.g., asterisks around a
word actuallylook like \*emphasis\*. Markdown lists look like, well, lists.
Evenblockquotes look like quoted passages of text, assuming you've everused
email.<h3 id="html">Inline HTML</h3>Markdown's syntax is intended for one
purpose: to be used as aformat for *writing* for the web.Markdown is
```

**fixed size 2 (128, 16):**
0.7603644
```for Markdown is to make it easy to read,
write, andedit prose. HTML is a *publishing* format; Markdown is a
*writing*format. T
```

**structure-aware:**
0.6797999
```* * *  <h2 id="overview">Overview</h2>  <h3
id="philosophy">Philosophy</h3>  Markdown is intended to be as easy-to-read and
easy-to-write as is feasible.  Readability, however, is emphasiz ed above all
else. A Markdown-formatted document should be publishable as-is, as plain text,
without looking like it's been marked up with tags or formatting instructions.
While Markdown's syntax has been inf luenced by several existing text-to-HTML
filters -- including [Setext][1], [atx][2], [Textile][3], [reStructuredText][4],
[Grutatext][5], and [EtText][6] -- the single biggest source of inspiration for
Markdo wn's syntax is the format of plain text email.
```

### Analysis
The similarity scores are radically higher in this test, and the best-match
answers are much 'better' at first glance.  Strangely, again, the most obvious
choice to the human eye is missed by the model, however it *has* selected far
better near-misses. This clearly implies to me that rather than getting close in
terms of meaning, the model is working purely based on numeric nearness of the
embedded vectors.  it is somewhat effective, but heavily flawed.

## semantic chunker vs all
The new semantic chunker seems to be radically more effective.  For our test
prompt with "what is markdown not a replacement for?" it finds the correct
match with a high similarity score:

0.80749995  `Markdown is not a replacement for HTML, or even close to it.`
0.7990001   `The idea for Markdown is to make it easy to read, write, and edit prose.`
0.72266996  `This means it's also easy to use Markdown to write about Markdown's own syntax.`
0.7101925   `HTML is a *publishing* format; Markdown is a *writing* format.`
0.695018    `Thus, Markdown's formatting syntax only addresses issues that can be conveyed in plain text.`

this approach to chunking appears to produce much more salient chunks at least
in the instance of this prompt.  Now to try another:

**prompt:** "are there any known bugs in markdown?"
this prompt should result in a high similarity match with a line in the test doc
that explicitly states that there are known bugs in markdown... yet it does not
rank this within the top 5 matches.  Worth noting is that the top match is a
very vague statement about markdown being easy to read and write.

**prompt:** "what type of lists does markdown support?"
this question is fairly specific, but there are numerous places in the doc that
would likely have highly related chunks.  the output does seem to corrobrate this
as matches 3 and 4 do mention the lists (the latter being the chunk I expected).
however, it ranks the aforementioned "easy to read and write" chunk in 2nd, and
an even more vague (but at least related) "markdown lists look like, well, lists"

## Analysis
from the looks of it, the sentence model and similarity test we are using here is
good at matching paraphrases and/or more general/common sentences, but do not do
well when asking for it to establish if a fact is true or not.  again this shows
that the 'meaning' similarity does not necessarily correlate with the symbol
similarity.
our third test also demonstrated that a neighbourhood of matching vectors can be
quite flat and result in very general answers ranking similarly to the few rare
closer matches

# BM25 best matching with sparse vectors for a hybrid approach
to improve the quality of our results, it is common to combine our dense chunking
approach with a best-match 25 approach.
bm25 scores query-to-chunk relevance across a document or collection of docs.  it
does this using statistics across all chunks.
first we tokenize our chunks, then we look at the frequency of terms within chunks.
we look at the frequency that "chunks containing the search term" occur within the
doc(s), and also look at an inverse of this.  "how much does a given chunk talk
about this specific term?" and "how rare is this term across all chunks?"

ultimately we use these values in our bm25 function in combination with two
parameters that control saturation - `k1` and `b`. `k1` is used to control the
term frequency, and `b` is used to normalise chunk lengths.

bm25:
```
BM25(q, d) =
Σ IDF(t) *
  ( TF(t,d) * (k1 + 1) ) /
  ( TF(t,d) + k1 * (1 - b + b * |d| / avgdl) )
```

description of bm25 equation:
```
sum over all query terms (tokens).  this is done independently per term
order/structure do not matter as a result

term_frequency(t, d) = number of times t occurs in chunk d
    this has limitations - eg longer chunks are more likely to score higher.
    repeated words may also affect the score (unless accounted for like we
    did)

term_frequency(t, d) * (k1 + 1) numerator
term_frequency(t, d) + k1 * (...) denominator
    these two ask "how impressed should i be by term repetition?"
    they allow TF to grow quickly at first but then flatten out

(...)
    1 - b + b * |d| / avgdl
        |d| = average number of tokens in THIS chunk
        avgdl = average chunk length across document(s)
        this is a normalisation that ensures long chunks don't just
        automatically score higher all the time

        b = 0 => ignore length entirely
        b = 1 => fully normalize score by length
        b = 0.75 => our test value which is balanced
        this helps us when chunk sizes are high variance

IDF(t)
    log((N - df + 0.5) / (df + 0.5) + 1)
        N = number of chunks
        df = number of chunks containing term t
        rare term = high IDF = strong signal
        common term = low IDF = weak
        "is this term discriminative?" and therefore important
```

when we combine bm25 and our dense embeddings, we are effectively combining:
"where is this explicitly mentioned?" (sparse)
"what is this about?" (dense)

## hybrid scoring test
**query:** "what type of lists does markdown support?"

**semantic chunk dense vector top 5**
0.861827   `Markdown lists look like, well, lists.`
0.6642482  `The idea for Markdown is to make it easy to read, write, and edit prose.`
0.65726805 `At some point in the future, Markdown may support starting ordered lists at an arbitrary number.`
0.6386822  `<h3 id="list">Lists</h3>  Markdown supports ordered (numbered) and unordered (bulleted) lists.`
0.6317893  `This means it's also easy to use Markdown to write about Markdown's own syntax.`

**BM25 top-5:**
10.500206354845922 `At some point in the future, Markdown may support starting ordered lists at an arbitrary number.`
7.15102861310484   `Markdown lists look like, well, lists.`
6.3583284784588034 `<h3 id="list">Lists</h3>  Markdown supports ordered (numbered) and unordered (bulleted) lists.`
5.718807300800917  `What a great season.`
5.718807300800917  `What a great season.`

**hybrid scores top 5**
0.8813225  `At some point in the future, Markdown may support starting ordered lists at an arbitrary number.`
0.8405185  `Markdown lists look like, well, lists.`
0.67331123 `<h3 id="list">Lists</h3>  Markdown supports ordered (numbered) and unordered (bulleted) lists.`
0.55348444 `Blockquotes can contain other Markdown elements, including headers, lists, and code blocks:  	> ## This is a header.`
0.5041493  `[1]: http://docutils.sourceforge.net/mirror/setext.html   [2]: http://www.aaronsw.com/2002/atx/   [3]: https://web.archi`

---

**query:** "known issues markdown"

**semantic chunk dense vector top 5**
0.5726969  `Markdown lists look like, well, lists.`
0.56948507 `The idea for Markdown is to make it easy to read, write, and edit prose.`
0.5640848  `[src]: /projects/markdown/syntax.text  * * *  <h2 id="overview">Overview</h2>  <h3 id="philosophy">Philosophy</h3>  Markdown is intended to be as easy-to-read and easy-to-write as is feasible.`
0.56243205 `Markdown is not a replacement for HTML, or even close to it.`
0.55363035 `Thus, Markdown's formatting syntax only addresses issues that can be conveyed in plain text.`

**BM25 top-5:**
7.0513037376018115 `Thus, Markdown's formatting syntax only addresses issues that can be conveyed in plain text.`
4.028387461592194  `The following three link definitions are equivalent:  	[foo]: http://example.com/  "Optional Title Here" 	[foo]: http://`
1.762926129404724  `This means it's also easy to use Markdown to write about Markdown's own syntax.`
1.6812931968446019 `Markdown lists look like, well, lists.`
1.6363255789224207 `Markdown: Syntax ================  <ul id="ProjectSubmenu">     <li><a href="/projects/markdown/" title="Markdown Projec`

**hybrid scores top 5**
0.98335373 `Thus, Markdown's formatting syntax only addresses issues that can be conveyed in plain text.`
0.6192186  `Markdown lists look like, well, lists.`
0.59827137 `The following three link definitions are equivalent:  	[foo]: http://example.com/  "Optional Title Here" 	[foo]: http://`
0.59649956 `[src]: /projects/markdown/syntax.text  * * *  <h2 id="overview">Overview</h2>  <h3 id="philosophy">Philosophy</h3>  Mark`
0.59435177 `Markdown is not a replacement for HTML, or even close to it.`

## Analysis
In our first test case here, both independent sets of scores had quite high
numbers.  in the case of the dense approach, this had actually resulted in a
very poor choice whereas the bm25 sparse approach performed well.  Likely this
is due to the size of chunks - the best match is a very short chunk so this
shows a bias toward short sentences which chunking in this manner. The hybrid
approach correctly combined the good match from bm25 and the third-best from
dense to show us the most valid answer.

In the second example, scores in the dense approach are much closer, in a
flatter and broader neighbourhood.  again, it points toward the short sentence
which is very generic and unhelpful, and it does not appear to find any
reasonably viable answers. bm25 and hybrid once again find the right answer
("the following three..." is truncated but contains the right text), though do
not this time rank it at the top.  changing the query slightly to "known bug in
markdown" - a direct quote - results in bm25 and hybrid finding the answer at
the top, but dense still misses

ultimately, the failures of the dense scoring approach are most likely explained
with the chunking method itself.  in the case of our second test, this seems
fairly obvious as the chunk containing the answer we want is quite large and
most likely loses 'meaning' in the scoring system.  it clearly favours shorter
sentences as it is now. dense ranking penalised the long chunk, and interpreted
the dominant meaning of it as something other than what we humans were
expecting.

## Chunking a real document
The document `sop1.txt` is copied from a google doc that was exported to HTML
format.  unfortunately this results in some odd formatting - and our use of 4
white spaces as a new additional separator for the sentence splitter (part of 
the semantic chunker).
Regardless, pressing on, we modify the files and can now run hybrid scoring
in other python modules.  Next came a simple script to take a query and a file
name (always `sop1.txt` in this case) and then do something with the scores
that are returned.
In this case, the top 3 (for brevity) are shown as usual.

## Results
**query:** "what is in the induction checklist?"
1.0000	`A copy of the completed induction checklist should be stored in the staff training records folder for inspection.`
0.7698	`Training records are kept in the Training Log.`
0.6541  `📁 MASTER F&B / KITCHEN EHO COMPLIANCE FOLDER(Hotel – UK) SECTION 1 — TRAINING & PEOPLE 1.1 Induction Checklist (New Sta`

**query:** "when must chemical spills be cleaned?"
1.0000  `Chemical spills cleaned immediately`
0.5946  `Surfaces must be cleaned and sanitised before preparation.`
0.5748  `Bins must be cleaned daily with appropriate disinfectant. `

As we can see, the chunking and scoring has worked fairly well here and given
some relevant matches.  The top matches both have a perfect `1.0`, with the
latter test having a larger gap before the second match.  clearly this simple
approach is incredibly confident of its answers - likely due to sentences and
chunks being quite small in general.

In the first example, the second best match has nothing to do with our
'induction checklist' however it does *appear* to contain a vaguely analogous
meaning.  This may be coincidence, it is difficult to tell, and it likely does
not seem to deserve the relatively high score.

In the second example, the second best match seems similar:  it is vaguely
related as it is about 'cleaning' but has nothing to do with 'chemical spills'.

in both cases, we have near-best-matches that are in a similar neighbourhood
semantically, but are not that relevant to our query.
To improve the accuracy/relevance of our results we would need to add further
complexity to our search:

- contextual filtering before ranking (here is where we should use the `meta`
attribute in our `Chunks`)
- sentence-level re-ranking inside top chunks.  split a chunk into sentences
and score the sentences vs the query, and only top sentences go to generator
- as always, be extra aware of the query and how it can have a significant
effect on the response

# Fixed token context
*NOTE: switched back to the test_doc that is about markdown.*

As opposed to using a top-N approach, we try using specific numbers of tokens.
We tested with 3 different lengths:  512, 1024, 2048.

To achieve this, we couldn't simply stick our chunks in until meeting a
token limit (eg splitting on spaces).  Instead we had to go through our
chunks and then use the sentencepiece tokenizer to properly tokenize the chunks
for gemma3.

for the same prompt that we used for top-n, we ended up getting the same answer
every time which was not particularly interesting:
`"Block-level HTML elements are allowed"`

i then changed the query to be a little more open ended:
`what does markdown allow us to do?`

in the case of 512 and 1024 length, we ended up with a completely unhelpful
response - repeating our query back to us.
However:  for 2048 length, we then got a very nice little summary of bullet
points that listed multiple relevant answers.

this led me to try the same query for top-N.  with 1, 3, or 5 top matches,
our responses were still pretty useless.  However, I noticed that the context
was much shorter.

I increased the N number to top 50 chunks to get a more similar context length.
Interestingly, the response was now much better - it was similar in meaning
to the fixed length answer, but different in word choice.

This suggests that in the case of the open-ended question, the larger context
window is actually more helpful as we can get a broader summation-style answer.
