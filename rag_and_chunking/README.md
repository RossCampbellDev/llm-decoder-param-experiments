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
