"""
Prompt templates for CoT-Evo variation module.

This module contains prompts for crossover and mutation operations.
Adapted from Supplementary_Material/prompt.py
"""

# ============================================================================
# Crossover Prompts
# ============================================================================

CROSSOVER_TEMPLATE = """<tips>Before formally considering the user's question, let me reiterate my responsibilities: deeply analyze the user's inquiry, and set breakpoints when necessary to append additional correct information for subsequent reasoning. Breakpoints will be wrapped with "<breakpoint>" and "</breakpoint>", containing key information required for follow-up analysis. I will not treat the content within breakpoints as prior knowledge, but will revalidate or re-explain them during remaining reasoning steps. I will ensure contextual continuity before and after breakpoints. I will use detailed and rich thinking after each breakpoint to ensure the accuracy and comprehensiveness of my reasoning.</tips>

{prefix}

<breakpoint>
I have received the following correct information that I should use to continue my reasoning:

{breakpoint}

I should mention all the information in my follow-up reasoning. I should revalidate or re-explain the above correct information (within the "<breakpoint>") in my reasoning later. I should continue the reasoning from the previous context.
</breakpoint>

"""

# Used to identify the binding point in a CoT (where errors start)
PREFIX_IDENTIFICATION_PROMPT = """You are a rigorous scientific evaluator. When facing scientific problems, you excel at judging whether the chain of thought for solving the problem is sufficiently correct and logical, and can keenly identify the point where errors occur but are no longer corrected. Given a problem, its corresponding chain of thought (CoT), and the correct answer, your task is to preserve the longest reasonable prefix from the CoT (i.e., the correct or logically sound skeleton) and delete the portion starting from where errors or deviations from reasonable reasoning occur. Therefore, you need to locate the first sentence that exhibits logical errors or significant deviation from correct reasoning and output that sentence exactly as it appears (ensuring it completely matches the content in the given thought, without rewriting or summarizing).

[Query Start]
{query}
[Query End]

[CoT Start]
{thought_current}
[CoT End]

[Answer Start]
{answer}
[Answer End]

### Instructions

You must strictly adhere to the following instructions:
1. Carefully read and analyze the entire reasoning trajectory and logic of the thought, ultilizing the correct answer to judge the correctness and reasonableness of each step.
2. After analyzing all steps, carefully determine from which sentence the reasoning begins to exhibit obvious errors and the subsequent reasoning trajectory significantly deviates from the correct reasoning path (i.e., no longer returns to a reasonable reasoning trajectory). This sentence is the first sentence that needs to be deleted.
3. Note that if errors occur during reasoning but are eventually corrected through reflection or other means later, it indicates that the subsequent reasoning trajectory has not deviated, and **such errors can be ignored**!
4. You must ensure that the extracted deletion sentence exactly matches the original thought word-for-word, without any rewriting.

### Output Format

1. You should first analyze the thought step by step and output the judgment of step accuracy to identify the reasoning prefix to be preserved and the first sentence to be deleted. If the final answer of the given CoT does not match the correct answer, you must find the first sentence that deviates from the correct reasoning path, because there has at least one error.
2. At the end of your output, you should output the first sentence to be deleted in the following format for easy answer extraction:
[RESULT_START]
First sentence to be deleted (exact word-for-word match with original, DO NOT OUTPUT other characters)
[RESULT_END]
"""

# Used to extract unique/correct information from a provider CoT
BREAKPOINT_EXTRACTION_PROMPT = """You are an insightful scientific reviewer, adept at discovering and extracting key and useful **knowledge or information**. Given a scientific problem, a chain of thought representing the model's current progress (CoT_current), a chain of thought from external exploration (CoT_external), and the corresponding correct answer, your task is to organize and extract the guiding information or knowledge in CoT_external that can help optimize or improve the reasoning trajectory of CoT_current, based on the correct answer. This valuable information will be used to guide CoT_current to optimize and modify its own thinking process to facilitate obtaining the correct answer.

[Query Start]
{query}
[Query End]

[CoT Current Start]
{thought_current}
[CoT Current End]

[CoT External Start]
{thought_external}
[CoT External End]

[Correct Answer Start]
{answer}
[Correct Answer End]

### Instructions

You must strictly follow these instructions:

1. The steps, information or final answer of CoT_external are partially incorrect. You should justify the correctness of them.
1. Carefully read and analyze CoT_external, and identify **all** the totally correct knowledge or information that are missing or incorrectly analyzed in CoT_current based on the correct answer. You need to ensure this information is useful to guide CoT_current to optimize or improve its reasoning trajectory.
2. Rewrite each knowledge or information extracted from CoT_external into a self-contained, generalizable sentence.
3. Ensure that the extracted information is completely correct (please judge based on the correct answer), genuinely exists, and only exists in CoT_external. Ensure that there are no contradictions among the extracted information!!!

### Output Format

1. You should first highlight the correct answer, and then output the analysis of CoT_current and CoT_external to identify useful knowledge or information in CoT_external that is missing or incorrectly analyzed in CoT_current.
2. Next, you should judge the correctness of each knowledge or information extracted from CoT_external based on the correct answer. You should delete the incorrect knowledge or information after the judgement.
3. At the end of the output, you should **output the remaining extracted correct knowledge or information as a list** according to the following format:

[RESULT_START]
* knowledge/information-1
* knowledge/information-2
* ...
[RESULT_END]
"""

# ============================================================================
# Mutation Prompts (Formula 12-14)
# ============================================================================

# Formula 12: Additive mutation - enrich with more details
MUTATION_ADD_PROMPT = """You are a reasoning expert rich in professional knowledge and good at thinking. Given a query and a Chain of Thought (CoT), your task is to add more details to the CoT to make the thinking process more complete, coherent, and of higher quality. You should ensure that the final CoT includes more than 800 words.

[Query Start]
{query}
[Query End]

[CoT Start]
{thought_current}
[CoT End]

### Instructions

You must strictly adhere to the following instructions:
1. Please read the CoT sentence by sentence and identify assertions, assumptions, or conclusions that lack substantial evidence, collectively referred to as "unverified elements".
2. Leverage your extensive knowledge to add more evidence and details in the context of these unverified elements, using the same writing style as the original text. Make sure that the final CoT includes more than 800 words!
3. Ensure that no characters in the reasoning chain are modified, only adding evidence and details.
4. Output the optimized CoT from the beginning.
5. You should first output "[RESULT_START]", then directly output the optimized CoT, and finally output "[RESULT_END]". Do not output other characters. The format is as follows:
[RESULT_START]
Optimized CoT, using the same format as the original CoT
[RESULT_END]
"""

# Formula 13: Deletive mutation - remove redundancy
MUTATION_DELETE_PROMPT = """You are an insightful scientific critic, adept at identifying unnecessary steps and unusual words or sentences in a complete scientific reasoning process. Given a scientific query and its corresponding Chain of Thought (CoT), your task is to identify the core skeleton of the reasoning trajectory and remove abrupt words and sentences, and steps that are completely unnecessary, meaningless, and do not advance the reasoning further. Finally, directly output the remaining complete, high-quality, and clear reasoning trajectory. You should ensure that every character you output truly exists in the original text.

[Query Start]
{query}
[Query End]

[CoT Start]
{thought_current}
[CoT End]

### Instructions

You must strictly adhere to the following instructions:
1. First, carefully read the CoT to find abrupt words and sentences, and redundant and meaningless steps that do not advance the reasoning further. DO NOT delete valuable exploratory steps, or necessary information and knowledge obtained and retrieved in the middle.
2. It should be noted that since the CoT may reference correct answers, tips, or additional information provided by the user, although these external aids are necessary, the sources of this information, such as "the user says" or "the tips mention", should not appear in the CoT. Please remove these information sources (e.g., 'user', 'tips', 'correct answer') and treat the information as the model's internal knowledge. If removing these sources causes contextual incoherence, **you can add some details to ensure coherence**.
3. Then, directly output the remaining complete, high-quality, and contextually coherent reasoning trajectory from the beginning. You should ensure that every character you output truly exists in the original text, i.e., you cannot modify any characters of the given CoT except for the deleted sentences. Use the original format of the CoT.
4. A complete reasoning trajectory may at least includes 4 core elements: rephrase the query (DO NOT delete the first few sentences), find the solution step by step, validate the solution, and summarize the final result.
5. You should output "[RESULT_START]" at the beginning of your response, then directly output the remaining reasoning trajectory, and output "[RESULT_END]" at the end of your response. The format is as follows:
[RESULT_START]
Remaining reasoning trajectory here, using the same format as the original CoT
[RESULT_END]
"""

# Part of Formula 14: Innovative mutation - diagnose errors using correct answer
MUTATION_INNOVATE_DIAGNOSE_PROMPT = """Given a query, a relevant chain of thought (CoT), and the correct answer, your task is to check the correctness of each step in the CoT based on the correct answer and find all the critical errors that occur in the CoT. Finally, write each error as a one-sentence advice to prevent the error from recurring.

[Query Start]
{query}
[Query End]

[CoT Start]
{thought_current}
[CoT End]

[Correct Answer Start]
{answer}
[Correct Answer End]

### Instructions

Please strictly follow the following instructions:
1. First, analyze the correctness of each step in the CoT based on the correct answer.
2. Then, identify all the critical errors that occur in the CoT. This includes using incorrect knowledge, making wrong inferences, or having faulty intuitions.
3. Summarize each error into a one-sentence advice to prevent the same errors from recurring in subsequent attempts. **DO NOT mention the correct answer in your summary.**
4. Output all the summarized errors in a list, strictly following the format below:
[RESULT_START]
* advice-1-summary
* advice-2-summary
* ...
[RESULT_END]
5. Please output "[RESULT_START]" first, then directly output all the summarized advices, and finally output "[RESULT_END]".
"""

# ============================================================================
# Helper for reasoning model guidance
# ============================================================================

REASONING_MODEL_GUIDANCE = """
Special instruction: You can receive externally provided information through <info></info>.
Upon receiving information, you should switch to a new thought and verify and use that information.
"""
