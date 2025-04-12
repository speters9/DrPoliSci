from dataclasses import dataclass

@dataclass
class Prompt:
    prompt: str
    rules: str
    guidelines: str
    examples: str
    json_format: str



eval_prompts = Prompt(
prompt="""
You are tasked with evaluating if a given context contains sufficient rich context to generate a fact-based question (factoid) and its answer. 

The evaluation should satisfy the rules below:
{rules}

Follow these steps to evaluate the context:
{guidelines}

Here are some examples (delimited by triple backticks):
```
{examples}
```
Now here is the context (delimited by triple quotes):

Context: \"\"\"{context}\"\"\" \n

Please use the JSON schema below for your output:
Output = {format}
Return Output
""",

rules= """ 
    - The context must present a clear subject or main idea.
    - The context must include specific details, facts, or examples.
    - The context must contain claims, arguments, or explanations that could be questioned.
    - The context must have sufficient depth or complexity to allow meaningful questions to be generated.
    """,

guidelines= """
    1. Read the context thoroughly to understand its depth and scope.
    2. Identify whether the context includes specific details or claims.
    3. Assess if a meaningful question can be generated from the information provided.
    4. Conclude if the context has "enough rich context" or "lacks sufficient context".
    """,

examples= """
    # Example 1:
    ## context: The Earth revolves around the Sun in an elliptical orbit, completing one revolution approximately every 365.25 days.
    ## reasoning": The context contains a clear subject (Earth's orbit) and provides specific details (elliptical orbit, 365.25 days).
    ## evaluation": Yes
    # Example 2:
    ## context": Apples are a type of fruit.
    ## reasoning": The context is too general and lacks specific details or claims to generate a meaningful question.
    ## evaluation": No

""",

json_format="""
    {
    "reasoning": "insert reasoning here",
    "evaluation": "<Yes / No>",
    }
"""
)


qa_prompts = Prompt(
    prompt= """
    You are tasked with generating a clear, fact-based question and its corresponding answer from the provided document context.

    The generated question and answer should follow these rules:
    {rules}

    Follow these steps to generate the question:
    {guidelines}

    Here are some examples (delimited by triple backticks):

    ```{examples}```

    Now here is the context (delimited by triple quotes):

    Context: \"\"\"{context}\"\"\"

    Please use the JSON schema below for your output:
    Output = {format}
    Return Output
    """,

    rules= """
        - The question must be fact-based and directly supported by the context.
        - The question should target specific details or claims found in the context.
        - The question must be clear, concise, and unambiguous.
        - The question should be formulated in the same style as questions users could ask in a search engine.
        - This means that your factoid question MUST NOT mention something like "according to the passage" or "context".
        - The answer must be accurate and directly derivable from the context.
        """,

    guidelines= """
        1. Read the provided context carefully to identify key facts or details.
        2. Formulate a question that tests comprehension of these details.
        3. Ensure the question is answerable using only the information in the context.
        4. Provide the correct answer based on the context.
        """,

    examples= """
    # Example 1:
    ## context: The Earth revolves around the Sun in an elliptical orbit, completing one revolution approximately every 365.25 days.
    ## generated question: What is the shape of Earth's orbit around the Sun?
    ## answer: Elliptical
    
    # Example 2:
    ## context: During photosynthesis, plants convert carbon dioxide and water into glucose and oxygen in the presence of sunlight.
    ## generated question: What happens during photosynthesis?
    ## answer: Plants use sunlight to convert carbon dioxide and water into glucose and oxygen.
    """,

    json_format= """
    {
    "question": "insert question here",
    "answer": "insert answer here",
    }
    """
)

@dataclass
class Task:
    name: str
    task: str
    evaluation_criteria: str
    evaluation_steps: str

@dataclass
class ValPrompt:
    prompt: str
    json_format: str

val_prompts = ValPrompt(
    prompt="""
    {task}
    {evaluation_criteria}

    Follow these steps to generate your evaluation:
    {evaluation_steps}

    Please respond using the following JSON schema:

    Answer = {format}

    You MUST provide values for 'Evaluation:' and 'Score' in your answer.

    Now here is the question (delimited by triple backticks)
    Question: ```{question}```

    Here is the context (delimited by triple quotes).
    Context: \"\"\"{context}\"\"\"\n

    Answer: """,

    json_format = """
    {
        "evaluation":"your rationale for the rating, as a text",
        "score":"your rating, as a number between 1 and 5",
    }
    """
)

groundedness = Task(
    name = "groundedness",
    task = """You will be given a context and a question.
        Your task is to evaluate the question based on the given context and provide a score between 1 and 5 according to the following criteria:
        """,

    evaluation_criteria = """
        - Score 1: The context does not provide sufficient information to answer the question in any way.
        - Score 2 or 3: The context provides some relevant information, but the question remains partially answerable, or is unclear/ambiguous.
        - Score 4: The context offers sufficient information to answer the question, but some minor details are missing or unclear.
        - Score 5: The context provides all necessary information to answer the question clearly and without ambiguity.
        """,

    evaluation_steps = """
        - Read the context and question carefully.
        - Analyse and evaluate the question based on the provided evaluation criteria.
        - Provide a scaled score between 1 and 5 that reflect your evaluation.
        """
)

relevance = Task(
    name = "relevance",
    task = """
            You will be provided with a question that may or may not relate to the roles of a student academic advisor (e.g. career advice, course recommendations, etc.).
            Your task is to evaluate its usefulness to users seeking information in the student academic advising domain and assign a score between 1 and 5 based on the following criteria:
            """,
    evaluation_criteria="""
        - Score 1: The question is unrelated to student academic advising.
        - Score 2 or 3: The question touches on student academic advising but leans more towards another domain and is not particularly useful or relevant for student academic advising-specific needs.
        - Score 4: The question is related to the student academic advising domain but lacks direct usefulness or relevance for users looking for valuable information in this domain.
        - Score 5: The question is clearly related to the student academic advising domain, makes sense, and is likely to be useful to users seeking information within this domain.
        """,
    evaluation_steps="""
        - Read the question carefully.
        - Analyse and evaluate the question based on the provided evaluation criteria.
        - Provide a scaled score between 1 and 5 that reflect your evaluation.
        """
    )

standalone = Task(
    name="standalone",
    task= """
        You will be given a question.
        Your task is to evaluate how context-independant this question is. You need to assess how self-contained and understandable a question is without relying on external context.
        The score reflects whether the question makes sense on its own. Questions referring to a specific, unstated context, such as "in the document" or "in the context," should receive a lower score. 
        Technical terms or acronyms related to student academic advising can still qualify for a high score if they are clear to someone with standard domain knowledge and documentation access.
        Please provide a score between 1 and 5 based on the following criteria:
        """,
    evaluation_criteria="""
        - Score 1: The question is highly dependent on external context and cannot be understood without additional information.
        - Score 2: The question provides some clarity but still requires significant additional context to make sense.
        - Score 3: The question can mostly be understood but may depend slightly on an external context for complete clarity.
        - Score 4: The question is nearly self-contained, with only minor reliance on external context.
        - Score 5: The question is entirely self-contained and makes complete sense on its own, without any reliance on external context.
        """,
    evaluation_steps="""
        - Read the question carefully.
        - Analyse and evaluate the question based on the provided evaluation criteria.
        - Provide a scaled score between 1 and 5 that reflect your evaluation.
        """
)

