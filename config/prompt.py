prompt_lbv1_cot="""
You are given a long document such as a story, meeting script, a news article, etc, and a question. Your task is to answer the question based on the information provided in the document. You should follow the instructions below to provide an accurate reasoning path, as well as a concise answer to the question:

**Instructions:**
Step 1. **Reasoning:** Imagine you are a student who has no prior knowledge about the giving context. Your task is to answer the questions based solely on the information presented here. First retrieve all relevant information, then deduce the correct answer. Begin by carefully reading the provided context. Identify and extract all relevant information that is directly related to the question. Be succinct and only extract the most important excerpts that will help you answer the question. Finally, deduce the correct answer based on the retrieved information.
Step 2. **Answer:** Using the information you have retrieved, and your deduction, answer the question as concisely as you can, using a single phrase or sentence if possible. Ensure that your answer should be brief and to the point.
Step 3. **Format Your Response:** Present your response in JSON format, comprising two components: "reasoning" and "answer". The "reasoning" section should detail your thought process, including the breakdown of the question, the relevant excerpts (indicated by [Excerpt xxx] at the start), and the derived conclusion. Ensure that each excerpt is an exact match to the original document. Limit the number of excerpts to a maximum of 10. The "answer" part should contain your final answer to the question, as concise and to the point as possible.

Illustrative Examples:

Example #1:

**Context:** [... Saltram is living with the Mulvilles at Wimbledon ... He is not working or producing anything ... He is idle and dependent on others ...]
**Question:** What is Saltram's living situation?

**Response:**
{{
    "reasoning": "Let me first retrieve relevant excerpts from the document, then answer the question. The question asks about Saltram's living situation. In the document, I can first locate that [Excerpt 1] `Saltram is living with the Mulvilles at Wimbledon`. Additionally, it is mentioned that [Excerpt 2] `He is not working or producing anything` and [Excerpt 3] `He is idle and dependent on others`. From these excerpts, I can deduce that Saltram is a guest in the home of the Mulvilles.",
    "answer": "He is a guest in the home of the Mulvilles."
}}

Example #2:

**Context:** [... The Collegian is the bi-weekly official student publication of Houston Baptist University in Houston, Texas ... Houston Baptist University, affiliated with the Baptist General Convention of Texas, offers bachelor's and graduate degrees. It was founded in 1960 ...]
**Question:** When was the institute that owned The Collegian founded?

**Response:**
{{
    "reasoning": "Let me first retrieve relevant excerpts from the document, then answer the question. The question asks about the founding date of the institute that owned The Collegian. In the document, I can first locate that [Excerpt 1] `The Collegian is the bi-weekly official student publication of Houston Baptist University in Houston, Texas`, so I need to look for information about Houston Baptist University. I find that [Excerpt 2] `Houston Baptist University was founded in 1960`. Therefore, the institute that owned The Collegian was founded in 1960.",
    "answer": "1960"
}}


Now, based on the context provided below, answer the question as concisely as you can, using a single phrase or sentence if possible. Meanwhile, reasoning must comply with the original text, and any knowledge should be derived from the original text.

**Context:** {context}
**Question:** {question}

**Response:**
"""

prompt_lbv1_nocot="""
You are given a long document such as a story, meeting script, a news article, etc, and a question. Your task is to answer the question based on the information provided in the document. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nContext:{context}\n\nNow, answer the question based on the context as concisely as you can, using a single phrase if possible. Do not provide any explanation and only give the best answer once.\n\nQuestion:{question}\n\nAnswer:"""


prompt_lbv2_cot="""
You are given a long document such as a story, meeting script, a news article, etc, and a question. Your task is to answer the question based on the information provided in the document. You should follow the instructions below to provide an accurate reasoning path, as well as a answer chosen from ABCD options to the question:

**Instructions:**
Step 1. **Reasoning:** First retrieve all relevant information, then deduce the correct answer. Begin by carefully reading the provided context. Identify and extract all relevant information that is directly related to the question. Be succinct and only extract the most important excerpts that will help you answer the question. Finally, deduce the correct answer based on the retrieved information.
Step 2. **Answer:** Using the information you have retrieved, and your deduction, answer the question as concisely as you can, using a single phrase or sentence if possible. Ensure that your answer should be brief and to the point.
Step 3. **Format Your Response:** Present your response in JSON format, comprising two components: "reasoning" and "answer". The "reasoning" section should detail your thought process, including the breakdown of the question, the relevant excerpts (indicated by [Excerpt xxx] at the start), and the derived conclusion. Ensure that each excerpt is an exact match to the original document. Limit the number of excerpts to a maximum of 10. The "answer" part should contain your final answer to the question, which is a choice selected from the ABCD options.

Illustrative Examples:

Example #1:

**Context:** [... Saltram is living with the Mulvilles at Wimbledon ... He is not working or producing anything ... He is idle and dependent on others ...]
**Question:** What is Saltram's living situation?
**Choices:**
(A) He is a guest in the home of the Mulvilles.
(B) He is in a hotel.
(C) He is homeless now.
(D) Unkonwn

**Response:**
{{
    "reasoning": "Let me first retrieve relevant excerpts from the document, then answer the question. The question asks about Saltram's living situation. In the document, I can first locate that [Excerpt 1] `Saltram is living with the Mulvilles at Wimbledon`. Additionally, it is mentioned that [Excerpt 2] `He is not working or producing anything` and [Excerpt 3] `He is idle and dependent on others`. From these excerpts, I can deduce that Saltram is a guest in the home of the Mulvilles.",
    "answer": "A"
}}

Now, based on the context provided below, answer the question with a choice selected from ABCD.

**Context:** {context}
**Question:** {question}
**Choices:**
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

**Response:**
"""

prompt_lbv2_nocot="""
Please read the following text and answer the question below.

{context}

What is the correct answer to this question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Format your response as follows: "The correct answer is (insert answer here)".
"""

prompt_cot="""Given a document and a question, answer concisely using a single phrase and provide a brief reasoning process. \n\nContext:{context}\n\n Now, answer the question based on the context as concisely as you can and give a reasoning paths, using a single phrase if possible. \n\nQuestion:{question}\n\n Format your response as:
Answer: []
Reasoning: []
Ensure both sections are separated clearly for easy extraction."""