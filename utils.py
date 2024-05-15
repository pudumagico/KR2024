import re
from pattern.text.en import singularize
import datetime
import random
import os
import litellm
import json
from openai import OpenAI
from litellm import completion
import clingo
from prompt.prompt_builder import PromptBuilder

from contextlib import redirect_stderr, nullcontext
import io
import logging
import sys

STDERR = ""


def new_stderr(old):
    def new(*args):
        # put your code here, you will intercept writes to stderr
        print("Intercepted: " + repr(args))
        global STDERR  # add new write to STDERR
        STDERR += args[0]
        old(*args)

    return new


sys.stderr.write = new_stderr(sys.stderr.write)


# Setup a logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("clingo_errors")

client = OpenAI(
    api_key="",
)
os.environ["OPENAI_API_KEY"] = ""
os.environ["COHERE_API_KEY"] = ""
os.environ["HUGGINGFACE_API_KEY"] = ""
os.environ["MISTRAL_API_KEY"] = ""
os.environ["ANYSCALE_API_KEY"] = ""
os.environ["VOYAGE_API_KEY"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

# set langfuse as a callback, litellm will send the data to langfuse
# litellm.success_callback = ["langfuse"]


def parse_error_callback(self, error, message):
    """
    Callback function to handle errors from Clingo.

    Parameters:
    error (Exception): The error object received from Clingo.

    Returns:
    str: A parsed error message.
    """

    self._data.get_logger().add_log("TEST", message)


def log_incumbent_theory(
    date_str,
    theory,
    model,
    max_retries,
    learning_examples,
    current_example,
    batch_examples,
):
    # Format the current date for the folder name
    # Create a folder name for this specific run
    run_folder_name = f"{model}_examples_{learning_examples}_batch{
        batch_examples}_retries{max_retries}"
    # Base directory for logs
    base_log_dir = "./logs"
    # Create the date-named directory
    date_dir = os.path.join(base_log_dir, date_str)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    # Create the run-specific directory
    run_dir = os.path.join(date_dir, run_folder_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    # Filename for the theory
    filename = f"current{current_example}_examples{learning_examples}.txt"
    # Full path for the file
    full_path = os.path.join(run_dir, filename)

    with open(full_path, "w") as log_file:
        log_file.write(theory)


def is_program_only_facts(program):
    """
    Check if the given ASP program only contains facts.
    """
    ctl = clingo.Control()
    ctl.add("base", [], program)

    try:
        ctl.ground([("base", [])])
    except clingo.ClingoError as e:
        print(f"Error in grounding: {e}")
        return False

    for atom in ctl.symbolic_atoms:
        if atom.is_fact:
            # Check if there are non-fact elements
            if not atom.symbol.arguments:
                return False
    return True


def logger(self, code: clingo.MessageCode, message: str) -> None:
    self._data.get_logger().add_log("TEST", message)


def check_asp_syntax(code):
    stderr_capture = io.StringIO()
    sys.stderr = stderr_capture
    null_device = io.StringIO()
    with redirect_stderr(null_device):
        try:

            # Setup logger
            errors = []

            def logger(code, message):
                # if code == clingo.WarningCode.RuntimeError:
                errors.append(message)

            ctl = clingo.Control(["--warn=none"], logger=logger)
            # ctl.configuration.logger = logger
            ctl.add("base", [], code)
            ctl.ground([("base", [])])
            return True, None
        except Exception as e:
            # error_message = parse_error_callback(e)
            # print(type(e))
            # print(e.args)
            # print('error', errors)
            # return logger.error(stderr_capture.getvalue())
            return False, errors[0]


# def run_asp_code(theory, examples):

#     null_device = io.StringIO()
#     correct = 0
#     with redirect_stderr(null_device):

#         for example in examples:

#             ctl = clingo.Control(["--warn=none"])
#             ctl.add("base", [], theory)
#             ctl.add("base", [], example['scene'])
#             ctl.add("base", [], example['question'])
#             ctl.add("base", [], "#show ans/1.")

#             ctl.ground([("base", [])])

#             with ctl.solve(yield_=True) as handle:
#                 for model in handle:
#                         # print(symbol.arguments[0], example['answer'], str(symbol.arguments[0]) == str(example['answer']))
#                         # exit()
#                         # if str(symbol.arguments[0]) == str(example['answer']):
#                     if answer_is_correct([s.arguments[0].name for s in model.symbols(shown=True)], example['answer']):
#                         correct += 1
#                     # else:
#                     #     print(example['question'])

#     incorrect = len(examples) - correct

#     # print('\n RUN ASP', correct, incorrect)
#     # if correct >= incorrect:
#     #     return True
#     # else:
#     #     return False

#     if not incorrect:
#         return True, None
#     else:
#         return False,


def check_asp_semantics(theory, examples):

    for example in examples:
        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], theory)
        ctl.add("base", [], example["scene"])
        ctl.add("base", [], example["question"])
        ctl.add("base", [], "#show ans/1.")
        ctl.ground([("base", [])])
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                if not answer_is_correct(
                    [s.arguments[0].name for s in model.symbols(shown=True)],
                    example["answer"],
                ):
                    return example
    return "Pass"


def run_asp_code(theory, examples):
    correct = 0
    incorrect_answers = []

    for example in examples:

        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], theory)
        ctl.add("base", [], example["scene"])
        ctl.add("base", [], example["question"])
        ctl.add("base", [], "#show ans/1.")

        ctl.ground([("base", [])])

        with ctl.solve(yield_=True) as handle:
            for model in handle:
                # print(symbol.arguments[0], example['answer'], str(symbol.arguments[0]) == str(example['answer']))
                # exit()
                # if str(symbol.arguments[0]) == str(example['answer']):
                if answer_is_correct(
                    [s.arguments[0].name for s in model.symbols(shown=True)],
                    example["answer"],
                ):
                    correct += 1
                else:
                    incorrect_answers.append(
                        [s.arguments[0].name for s in model.symbols(
                            shown=True)]
                    )

    incorrect = len(examples) - correct

    # print('\n RUN ASP', correct, incorrect)
    # if correct >= incorrect:
    #     return True
    # else:
    #     return False

    if not incorrect:
        return True, None
    else:
        return False, incorrect_answers


def mend_syntax(broken_rule, syntax_error, preprompt, model="gpt-4-turbo"):
    syntax_error = syntax_error.replace('[#inc_base];', '')
    print(syntax_error)
    preprompt = f"You must repair the syntax of the prompted Answer Set Programming rule(s). Additionally, Clingo outputed the following error: {
        syntax_error}. You must only output the fixed ASP rule(s) and any other rules included in the prompt that have correct syntax. Do not output any natural language. The output must be in plain text only! Do not output the response as a code block!"
    messages = [
        {"role": "system", "content": preprompt},
        {"role": "user", "content": broken_rule},
    ]
    response = completion(model=model, messages=messages, temperature=0)

    return response.choices[0].message.content


def mend_semantics(
    broken_rule, infered_answer, expected_answer, theory, preprompt, model="gpt-4-turbo"
):
    preprompt = f"""You must repair the semantics of the prompted Answer Set Programming rule(s).
    The original theory is: {theory}. The prompted rules are added to this theory to calculate the answer. 
    They do not calculate the correct answer, which is: {expected_answer}, 
    but instead result in the following incorrect answer: {infered_answer} (can me empty).
    You must only output the fixed ASP rule(s). 
    The rule(s) must be semantically different from the ones prompted and they should now calculate the correct answer.
    Do not output any natural language. The output must be in plain text only! Do not output the response as a code block!"""

    messages = [
        {"role": "system", "content": preprompt},
        {"role": "user", "content": broken_rule},
    ]
    response = completion(model=model, messages=messages, temperature=0)

    return response.choices[0].message.content

def mend_semantics_with_states(
    broken_rule, infered_answer, expected_answer, theory, preprompt, model="gpt-4-turbo"
):
    preprompt = f"""You must repair the semantics of the prompted Answer Set Programming rule(s).
    The original theory is: {theory}. The prompted rules are added to this theory to calculate the answer. 
    They do not calculate the correct answer, which is: {expected_answer}, 
    but instead result in the following incorrect answer: {infered_answer} (can me empty).
    Additionally, we give you the state the computation failed.
    You must only output the fixed ASP rule(s). 
    The rule(s) must be semantically different from the ones prompted and they should now calculate the correct answer.
    Do not output any natural language. The output must be in plain text only! Do not output the response as a code block!"""

    messages = [
        {"role": "system", "content": preprompt},
        {"role": "user", "content": broken_rule},
    ]
    response = completion(model=model, messages=messages, temperature=0)

    return response.choices[0].message.content

def ask_LLM(question, preprompt, model="gpt-4-turbo"):
    messages = [
        {"role": "system", "content": preprompt},
        {"role": "user", "content": question},
    ]
    response = completion(model=model, messages=messages, temperature=0)

    return response.choices[0].message.content


# def sanitize(llm_output):
#     if llm_output.choices:
#         sanitized = llm_output.choices[0].message.content

#     return sanitized

# def ask_LLM(question, preprompt, model = "gpt-3.5-turbo"):
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {"role": "system", "content":preprompt},
#             {"role": "user", "content": question},
#         ],
#         model="gpt-3.5-turbo"
#     )
#     return chat_completion.choices[0].message.content.strip('```').strip('{').strip('}').strip('asp').strip('ASP')


def merge_asp_encodings(encoding1, encoding2, model="gpt-3.5-turbo"):
    # Construct the prompt for GPT
    prompt = (
        "Here are two ASP encodings:\n\n"
        "Encoding 1:\n" + encoding1 + "\n\n"
        "Encoding 2:\n" + encoding2 + "\n\n"
        "Merge these two ASP encodings into a single coherent encoding."
    )

    # Send the prompt to GPT
    messages = [
        {
            "role": "system",
            "content": "Your task is to merge these two Answer Set Programming encodings. \
                    Remove any rule with constants in the head.\
                    Remove any redundant rules or non general rules (rules with constants in the head). \
                    RETURN ONLY ONE ASP ENCODING. NO NATURAL LANGUAGE EXPLANATIONS. JUST OUTPUT ASP.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = completion(model=model, messages=messages)

    # Return the merged encoding
    return (
        response.choices[0]
        .message.content.strip("```")
        .strip("{")
        .strip("}")
        .strip("asp")
        .strip("ASP")
        .strip("prolog")
    )


# def merge_asp_encodings(encoding1, encoding2):
#     # Construct the prompt for GPT
#     prompt = (
#         "Here are two ASP encodings:\n\n"
#         "Encoding 1:\n" + encoding1 + "\n\n"
#         "Encoding 2:\n" + encoding2 + "\n\n"
#         "Merge these two ASP encodings into a single coherent encoding."
#     )

#     # Send the prompt to GPT
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {"role": "system",
#              "content": "Your task is to merge these two Answer Set Programming encodings. \
#                         Remove any rule with constants in the head.\
#                         Remove any redundant rules or non general rules (rules with constants in the head). \
#                         RETURN ONLY ONE ASP ENCODING. NO NATURAL LANGUAGE EXPLANATIONS. JUST OUTPUT ASP."},
#             {
#                 "role": "user",
#                 "content": prompt,
#             }
#         ],
#         model="gpt-3.5-turbo"
#     )

#     # Return the merged encoding
#     return chat_completion.choices[0].message.content


def answer_is_correct(answers, correct_answer):
    correct = False

    for answer in answers:
        if answer == sanitize_asp(correct_answer):
            correct = True
        elif (
            (answer == "to_the_right_of" and correct_answer == "right")
            or (answer == "to_the_left_of" and correct_answer == "left")
            or (answer == "in_front_of" and correct_answer == "front")
        ):
            correct = True
    return correct


def sanitize(name):
    # source: DFOL-VQA
    plurale_tantum = [
        "this",
        "yes",
        "pants",
        "shorts",
        "glasses",
        "scissors",
        "panties",
        "trousers",
        "binoculars",
        "pliers",
        "tongs",
        "tweezers",
        "forceps",
        "goggles",
        "jeans",
        "tights",
        "leggings",
        "chaps",
        "boxers",
        "indoors",
        "outdoors",
        "bus",
        "octapus",
        "waitress",
        "pasta",
        "pita",
        "glass",
        "asparagus",
        "hummus",
        "dress",
        "cafeteria",
        "grass",
        "class",
    ]

    irregulars = {
        "shelves": "shelf",
        "bookshelves": "bookshelf",
        "olives": "olive",
        "brownies": "brownie",
        "cookies": "cookie",
    }

    temp = name.strip().lower()
    if temp in irregulars:
        return irregulars[temp]
    elif not (temp.split(" ")[-1] in plurale_tantum or temp[-2:] == "ss"):
        return singularize(temp)
    else:
        return temp


def cleanup_whitespace(name):
    cleanup_regex = r"[^\w]"
    return re.sub(cleanup_regex, "_", name)


def sanitize_asp(name):
    return cleanup_whitespace(sanitize(name))


def remove_random_lines(theory, percentage):
    # Read the content of the file
    lines = theory.split("\n")

    # Calculate the number of lines to remove
    num_lines_to_remove = int(len(lines) * (percentage / 100))

    # Randomly select lines to remove
    lines_to_remove = set(random.sample(
        range(len(lines)), num_lines_to_remove))

    # Create a new list of lines that does not include the lines to remove
    new_lines = [
        line + "\n" for i, line in enumerate(lines) if i not in lines_to_remove
    ]

    # Return the modified content
    return "".join(new_lines)


def remove_lines_with_predicates(input_string, predicates):
    # Split the string into lines
    lines = input_string.split("\n")

    # Filter out lines that contain any of the predicates
    filtered_lines = [
        line for line in lines if not any(predicate in line for predicate in predicates)
    ]

    # Join the remaining lines back into a single string
    return "\n".join(filtered_lines)


def write_to_log(output_file_name, *args):
    # Open the file in append mode
    with open(output_file_name, "a") as file:
        # Write each argument to the file, except for the first one
        for arg in args:
            file.write(str(arg) + "\n")


def save_examples(filename, array):
    # Open the file in write mode
    with open(filename, "w") as file:
        # Use json.dump to write the array to the file
        json.dump(array, file)
