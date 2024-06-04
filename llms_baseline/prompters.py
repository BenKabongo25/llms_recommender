# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Basline approach
# Prompts/formats for source and target

from typing import *
from common.utils.preprocess_text import preprocess_text


class SourcePrompter:
    """
    Args:
        - n_samples: int
            0 for zero-shot, >= 1 for few-shot
        - min_rating: int
        - max_rating: int
        - user_description_flag: bool
        - item_description_flag: bool
        - source_review_flag: bool
        - source_rating_flag: bool
        - user_first_flag: bool
        - user_only_flag : bool
        - target_review_flag: bool
        - target_rating_flag: bool
    """

    def __init__(self, args):
        self.args = args


    def _user_preprompt(self) -> str:
        if self.args.item_description_flag:
            if self.args.source_review_flag and self.args.source_rating_flag:
                prompt = (
                    "Here are some examples of user reviews and ratings for other items, "
                    "in the format item description | review | rating "
                    f"(from {self.args.min_rating} to {self.args.max_rating}):"
                )

            elif self.args.source_review_flag:
                prompt = (
                    "Here are some examples of user reviews for other items, "
                    "in the format item description | review:"
                )

            elif self.args.source_rating_flag:
                prompt =  (
                    "Here are some examples of user ratings for other items, "
                    "in the format item description | rating "
                    f"(from {self.args.min_rating} to {self.args.max_rating}):"
                )

            else:
                prompt = (
                    "Here are some descriptions of other items with which "
                    "the user has interacted:"
                )

        elif self.args.source_review_flag:
            if self.args.source_rating_flag:
                prompt = (
                    "Here are some examples of user reviews and ratings for other items, "
                    "in the format review | rating "
                    f"(from {self.args.min_rating} to {self.args.max_rating}):"
                )

            else:
                prompt = "Here are some examples of user reviews for other items:"

        elif self.args.source_rating_flag:
            prompt = (
                "Here are some examples of user ratings "
                f"(from {self.args.min_rating} to {self.args.max_rating}) for other items:"
            )

        else:
            prompt = ""

        return prompt


    def _item_preprompt(self) -> str:
        if self.args.user_description_flag:
            if self.args.source_review_flag and self.args.source_rating_flag:
                prompt = (
                    "Here are some examples of other users' reviews and ratings for the item, "
                    "in the format user description | review | rating "
                    f"(from {self.args.min_rating} to {self.args.max_rating}):"
                )

            elif self.args.source_review_flag:
                prompt = (
                    "Here are some examples of other users' reviews for the item, "
                    "in the format user description | review:"
                )

            elif self.args.source_rating_flag:
                prompt =  (
                    "Here are some examples of other users' ratings for the item, "
                    "in the format user description | rating "
                    f"(from {self.args.min_rating} to {self.args.max_rating}):"
                )

            else:
                prompt = (
                   " Here are some descriptions of other users who have "
                   "interacted with the item:"
                )

        elif self.args.source_review_flag:
            if self.args.source_rating_flag:
                prompt = (
                    "Here are some examples of other users' reviews and ratings for the item, "
                    "in the format review | rating "
                    f"(from {self.args.min_rating} to {self.args.max_rating}):"
                )

            else:
                prompt = "Here are some examples of other users' reviews for the item:"

        elif self.args.source_rating_flag:
            prompt = (
                "Here are some examples of other users' ratings "
                f"(from {self.args.min_rating} to {self.args.max_rating}) for the item"
            )

        else:
            prompt = ""

        return prompt


    def _task_prompt(self) -> str:
        if self.args.target_review_flag and self.args.target_rating_flag:
            task_prompt = "What are the user's review and rating for the item? In review | rating format."
        elif self.args.target_review_flag:
            task_prompt = "What is the user's textual review of the item?"
        elif self.args.target_rating_flag:
            task_prompt = f"What is the user's rating of the item?"
        else:
            task_prompt = ""
        return task_prompt


    def zero_shot_prompt(self, sample: dict) -> str:
        user_examples = sample["user_examples"]
        user_prompt = self._user_preprompt() + "\n" + user_examples + "\n\n"

        item_prompt = ""
        if not self.args.user_only_flag:
            item_examples = sample["item_examples"]
            item_prompt = self._item_preprompt() + "\n" + item_examples + "\n\n"
        
        prompt = "Let's consider an user-item pair.\n\n"
        if self.args.user_first_flag:
            prompt += user_prompt + item_prompt
        else:
            prompt += item_prompt + user_prompt
        
        user_description = sample["user_description"]
        if self.args.user_description_flag and len(user_description) > 0:
            prompt += "The user description is " + user_description
            if not prompt.endswith("."):
                prompt += "."
            prompt += " "
                
        item_description = sample["item_description"]
        if self.args.item_description_flag and len(item_description) > 0:
            prompt += "The item description is " + item_description
            if not prompt.endswith("."):
                prompt += "."
            prompt += " "

        task_prompt = self._task_prompt()
        prompt += task_prompt

        return prompt
    

    def _few_preprompt(self) -> str:
        prompt = ""
        
        if self.args.target_review_flag and self.args.target_rating_flag:
            prompt += "You have to predict the review and the rating for each user-item pair. "

        elif self.args.target_review_flag:
            prompt += "You have to predict the review for each user-item pair. "

        elif self.args.target_rating_flag:
            prompt += "You have to predict the review for each user-item pair. "

        else:
            prompt += ""

        if self.args.item_description_flag:
            if self.args.source_review_flag and self.args.source_rating_flag:
                prompt += (
                    "For each pair, you'll base your predictions on the user's interactions "
                    "with other items, which are in the format item description | review | rating"
                    f"(from {self.args.min_rating} to {self.args.max_rating})."
                )
                    
            elif self.args.source_review_flag:
                prompt += (
                    "For each pair, you'll base your predictions on the user's interactions "
                    "with other items, which are in the format item description | review."
                )

            elif self.args.source_rating_flag:
                prompt += (
                    "For each pair, you'll base your predictions on the user's interactions "
                    "with other items, which are in the format item description | rating"
                    f"(from {self.args.min_rating} to {self.args.max_rating})."
                )

            else:
                prompt += ""

        elif self.args.source_review_flag:
            if self.args.source_rating_flag:
                prompt += (
                    "For each pair, you'll base your predictions on the user's "
                    f"reviews and ratings (from {self.args.min_rating} to {self.args.max_rating}) "
                    "for other items."
                )

            else:
                prompt += (
                    "For each pair, you'll base your predictions on the user's "
                    "reviews for other items."
                )

        elif self.args.source_rating_flag:
            prompt += (
                "For each pair, you'll base your predictions on the user's "
                f"ratings (from {self.args.min_rating} to {self.args.max_rating}) "
                "for other items."
            )

        else:
            prompt += ""

        if not self.args.user_only_flag:
            prompt += " "

            if self.args.user_description_flag:
                if self.args.source_review_flag and self.args.source_rating_flag:
                    prompt += (
                        "You'll also base your predictions on the item's interactions "
                        "with other users, which are in the format user description | review | rating"
                        f"(from {self.args.min_rating} to {self.args.max_rating})."
                    )

                elif self.args.source_review_flag:
                    prompt += (
                        "You'll also base your predictions on the item's interactions "
                        "with other users, which are in the format user description | review."
                    )

                elif self.args.source_rating_flag:
                    prompt += (
                        "You'll also base your predictions on the item's interactions "
                        "with other users, which are in the format user description | rating"
                        f"(from {self.args.min_rating} to {self.args.max_rating})."
                    )

                else:
                    prompt += ""

            elif self.args.source_review_flag:
                if self.args.source_rating_flag:
                    prompt += (
                        "You'll also base your predictions on the item's "
                        f"reviews and ratings (from {self.args.min_rating} to {self.args.max_rating}) "
                        "from other users."
                    )

                else:
                    prompt += (
                        "You'll also base your predictions on the item's "
                        "reviews from other users."
                    )

            elif self.args.source_rating_flag:
                prompt += (
                    "You'll also base your predictions on the item's "
                    f"ratings (from {self.args.min_rating} to {self.args.max_rating}) "
                    "from other users."
                )

            else:
                prompt = ""

        return prompt
    

    def few_shot_prompt(self, sample: dict) -> str:
        prompt = self._few_preprompt()

        if self.args.target_review_flag and self.args.target_rating_flag:
            task_prompt = "Review and rating"
        elif self.args.target_review_flag:
            task_prompt = "Review"
        elif self.args.target_rating_flag:
            task_prompt = "Rating"
        else:
            task_prompt = ""
    
        shots_samples = sample["shots"]
        ui_sample = sample["sample"]

        def _format(sample, target_flag=True):
            user_prompt = ""
            user_description = sample["user_description"]
            if self.args.user_description_flag and len(user_description) > 0:
                user_prompt += "User description: " + user_description + "\n"
            user_prompt += "User interactions: " + sample["user_examples"] + "\n"

            item_prompt = ""
            if not self.args.user_only_flag:
                item_description = sample["item_description"]
                if self.args.item_description_flag and len(item_description) > 0:
                    item_prompt += "Item description: " + item_description + "\n"
                item_prompt += "Item interactions: " + sample["item_examples"] + "\n"
            
            if self.args.user_first_flag:
                sample_prompt = user_prompt + item_prompt
            else:
                sample_prompt = item_prompt + user_prompt

            sample_prompt += task_prompt + ": "
            if target_flag:
                if self.args.target_review_flag and self.args.target_rating_flag:
                    sample_prompt += sample["review"] + "|" + sample["rating"]
                elif self.args.target_review_flag:
                    sample_prompt += sample["review"]
                elif self.args.target_rating_flag:
                    sample_prompt += sample["rating"]
                else:
                    sample_prompt += ""

            return sample_prompt

        for shot_sample in shots_samples:
            prompt += "\n\n"
            prompt += _format(shot_sample, target_flag=True)

        prompt += "\n\n"
        prompt += _format(ui_sample, target_flag=False)

        return prompt
    
    
    def prompt(self, sample):
        if self.args.n_samples == 0:
            return self.zero_shot_prompt(sample)
        return self.few_shot_prompt(sample)


class TargetFormer:
    """
    Args:
        - target_review_flag: bool
        - target_rating_flag: bool
    """

    def __init__(self, args):
        self.args = args

    def format(self, sample):
        review, rating = sample["review"], sample["rating"]
        if self.args.target_review_flag and self.args.target_rating_flag:
            target = f"{review} | {rating}"
        elif self.args.target_review_flag:
            target = review
        elif self.args.target_rating_flag:
            target = rating
        else:
            target = ""
        return target
    
    @classmethod
    def get_review_rating(cls, output: str) -> tuple:
        split = output.split("|")
        review = split[0]
        rating = ""
        if len(split) > 1:
            rating = split[1].strip()
        return review, rating
    
    @classmethod
    def process_text(cls, text: str, max_length: int, args: Any) -> str:
        text = preprocess_text(text, args)
        if args.truncate_flag:
            text = str(text).strip().split()
            if len(text) > max_length:
                text = text[:max_length - 1] + ["..."]
        return " ".join(text) 
