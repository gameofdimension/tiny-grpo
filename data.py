SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_answer_from_model_output(text):
   """
   Extracts the value from the last <answer> tag in the text.

   Args:
       text (str): The model-generated text containing XML-style <answer> tags.

   Returns:
       str or None: The content inside the <answer> tags, or None if no valid answer is found.

   Explanation:
       1. Splits the text on the <answer> tag to isolate content after the tag.
       2. Checks if at least one <answer> tag exists in the text.
       3. For the last <answer> segment:
          - Verifies it contains a closing </answer> tag.
          - Extracts only the content between the tags.
       4. Returns None if the answer is empty (just "...") or if tags are missing.
   """
   # Split on <answer> and take everything after the last occurrence
   parts = text.split("<answer>")
   if len(parts) < 2:  # No <answer> tag found
       return None
   last_part = parts[-1]

   # Extract content up to </answer>
   if "</answer>" not in last_part:
       return None
   answer = last_part.split("</answer>")[0].strip()
   return None if answer == "..." else answer

def extract_answer_from_dataset(text):
   """
   Extracts the answer from the GSM8K dataset examples.

   Args:
       text (str): The dataset example text containing a question and answer.

   Returns:
       str or None: The extracted answer part after the '####' delimiter, or None if not found.

   Explanation:
       1. Checks if the text contains the '####' delimiter that separates question from answer.
       2. If found, splits the text at this delimiter and returns the second part (the answer).
       3. The answer is stripped of leading/trailing whitespace.
       4. Returns None if no delimiter is present.
   """
   if "####" not in text:
       return None
   return text.split("####")[1].strip()


def prepare_dataset(split="train"):
   """
   Load and prepare the GSM8K dataset for training with string prompts.

   Args:
       split (str): The dataset split to load ("train" or "test"). Defaults to "train".

   Returns:
       list: A list of formatted examples, each containing a prompt string and answer.

   Explanation:
       1. Loads the GSM8K dataset from the Hugging Face datasets hub.
       2. For each example in the dataset:
          - Creates a list of messages with system prompt and the question.
          - Converts this list into a single string prompt using build_prompt().
          - Extracts the answer from the dataset example.
          - Creates a formatted example dictionary with prompt and answer.
       3. Returns the list of formatted examples ready for model training or evaluation.
   """
   data = load_dataset('openai/gsm8k', 'main')[split]
   formatted_data = []
   for example in data:
       # Convert list of messages to a single string prompt.
       prompt_str = build_prompt([
           {"role": "system", "content": SYSTEM_PROMPT},
           {"role": "user", "content": example["question"]}
       ])
       formatted_example = {
           "prompt": prompt_str,  # Now a string rather than a list.
           "answer": extract_answer_from_dataset(example["answer"])
       }
       formatted_data.append(formatted_example)
   return formatted_data

def build_prompt(messages):
   """
   Build a single prompt string from a list of messages.

   Args:
       messages (list): A list of message dictionaries, each with 'role' and 'content' keys.

   Returns:
       str: A concatenated string of all message contents.

   Explanation:
       1. Takes a list of message dictionaries in the typical chat format.
       2. Extracts the 'content' field from each message and strips whitespace.
       3. Joins all content strings with newlines to create a single prompt.
       4. This preserves the training format while converting from structured messages to a string.
   """
   return "\n".join([msg["content"].strip() for msg in messages])