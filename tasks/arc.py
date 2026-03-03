"""
The ARC dataset from Allen AI.
https://huggingface.co/datasets/allenai/ai2_arc
"""

import os
from datasets import load_dataset
from tasks.common import Task, render_mc
from nanochat.common import get_global_config

class ARC(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC subset must be ARC-Easy or ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC split must be train|validation|test"
        local_dir = get_global_config().allenai_arc_dataset
        if local_dir and os.path.isdir(local_dir):
            print(f"Loading ARC dataset from local directory {local_dir}...")
            self.ds = load_dataset(local_dir, subset, split=split).shuffle(seed=42)
        else:
            print(f"Loading ARC dataset from Hugging Face Hub...")
            self.ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"] # the question text
        choices = row["choices"]["text"] # the text of each choice
        answer_string = row["answerKey"] # e.g. "A", "B", "C", "D"
        letters = row["choices"]["label"] # e.g. ["A", "B", "C", "D"]
        assert answer_string in letters, f"ARC answer {answer_string} must be one of {letters}" # sanity check
        # create and return the Conversation object
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        conversation = {
            "messages": messages,
            "letters": letters, # useful during evaluation, so we can narrow and clamp the assistant prediction to one of the letters
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        # the assert here is not strictly speaking needed, but currently the way we eval, we expect this to be true
        # I'm going to leave the assert here to prevent footguns, but possibly in the future can remove it.
        assert assistant_response in conversation['letters'], f"ARC answer {assistant_response} is expected to be one of {conversation['letters']}"
        assistant_message = conversation['messages'][-1]['content'] # e.g. "A"
        return assistant_response == assistant_message

if __name__ == "__main__":
    arc = ARC(subset="ARC-Easy", split="train")
    print(arc.get_example(0))
    print(arc.evaluate(arc.get_example(0), "A"))
    print(arc.evaluate(arc.get_example(0), "B"))
    print(arc.evaluate(arc.get_example(0), "C"))
    print(arc.evaluate(arc.get_example(0), "D"))