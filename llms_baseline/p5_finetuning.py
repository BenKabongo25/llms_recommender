# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# P5: https://arxiv.org/abs/2203.13366
# Fine-tuning and evaluation

import enum
from t5_conditional_generation import *


class PromptType(enum.Enum):
    RATING_1 = 0
    RATING_2 = 1
    REVIEW = 2


class SourcePrompter:

    def __init__(self, args):
        self.args = args
        self.prompt_type = PromptType(args.prompt_type)

    def prompt(self, sample: dict) -> str:
        user_id = sample[self.args.user_id_column]
        item_id = sample[self.args.item_id_column]
        item_description = sample[self.args.item_description_column]

        if self.prompt_type is PromptType.RATING_1:
            text = (
                f"Which star rating will user_[{user_id}] give item_[{item_id}]? "
                f"(1 being lowest and 5 being highest)"
            )
        elif self.prompt_type is PromptType.RATING_2:
            text = (
                f"How will user_[{user_id}] rate this product: [{item_description}]? "
                f"(1 being lowest and 5 being highest)"
            )
        elif self.prompt_type is PromptType.REVIEW:
            text = (
                f"Generate an explanation for user_[{user_id}] about this product: [{item_description}]"
            )
        else:
            text = ""
        
        return text


class TargetFormer:

    def __init__(self, args):
        self.args = args
        self.prompt_type = PromptType(args.prompt_type)

    def target(self, sample: dict) -> str:
        if self.prompt_type is PromptType.RATING_1:
            target = sample[self.args.rating_column]
        elif self.prompt_type is PromptType.RATING_2:
            target = sample[self.args.rating_column]
        elif self.prompt_type is PromptType.REVIEW:
            target = preprocess_text(
                text=sample[self.args.review_column], 
                args=self.args,
                max_length=self.args.max_target_length
            )
        else:
            target = ""
        
        return str(target)


class P5DataCreator:

    def __init__(self, args):
        self.args = args
        self.source_prompter = SourcePrompter(args)
        self.target_former = TargetFormer(args)

    def create_dataset(
        self, 
        data_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame]=None,
        items_df: Optional[pd.DataFrame]=None
    ) -> pd.DataFrame: 
        data = []
        for i, sample in tqdm(
            data_df.iterrows(),
            desc="Dataset creation",
            colour="green",
            total=len(data_df)
        ):
            sample = sample.to_dict()
            item_description = ""
            if items_df is not None:
                descs = items_df[
                    items_df[self.args.item_id_column] == sample[self.args.item_id_column]
                ][self.args.item_description_column].values
                if len(descs) > 0:
                    item_description = descs[0]
                    item_description = preprocess_text(
                        text=item_description, 
                        args=self.args,
                        max_length=self.args.max_description_length
                    )
            sample[self.args.item_description_column] = item_description

            source = self.source_prompter.prompt(sample)
            target = self.target_former.target(sample)
            data.append({
                "user_id": sample[self.args.user_id_column],
                "item_id": sample[self.args.item_id_column],
                "source": source,
                "target": target
            })
        p5_data_df = pd.DataFrame(data)
        return p5_data_df
        

def get_train_test_data(args: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if args.train_text_data_path != "" and args.test_text_data_path != "":
        train_df = pd.read_csv(args.train_text_data_path)
        test_df = pd.read_csv(args.test_text_data_path)
    
    else:
        if args.dataset_path == "" and (args.train_dataset_path == "" or args.test_dataset_path == ""):
            seen_dir = os.path.join(args.dataset_dir, "samples", "splits", "seen")
            args.train_dataset_path = os.path.join(seen_dir, "train.csv")
            args.test_dataset_path = os.path.join(seen_dir, "test.csv")

        if args.train_dataset_path != "" and args.test_dataset_path != "":
            train_data_df = pd.read_csv(args.train_dataset_path)
            test_data_df = pd.read_csv(args.test_dataset_path)
        
        else:
            data_df = pd.read_csv(args.dataset_path)
            train_data_df = data_df.sample(frac=args.train_size, random_state=args.random_state)
            test_data_df = data_df.drop(train_data_df.index)

        metadata_dir = os.path.join(args.dataset_dir, "samples", "metadata")
        users_df = None
        if args.user_description_flag:
            if args.users_path == "":
                args.users_path = os.path.join(metadata_dir, "users.csv")
            users_df = pd.read_csv(args.users_path)

        items_df = None
        if args.item_description_flag:
            if args.items_path == "":
                args.items_path = os.path.join(metadata_dir, "items.csv")
            items_df = pd.read_csv(args.items_path)

        p5_data_creator = P5DataCreator(args)
        train_df = p5_data_creator.create_dataset(
            train_data_df,
            users_df=users_df,
            items_df=items_df
        )
        test_df = p5_data_creator.create_dataset(
            test_data_df,
            users_df=users_df,
            items_df=items_df
        )

        if args.save_data_flag:
            if args.save_data_dir == "":
                args.save_data_dir = os.path.join(args.dataset_dir, "samples", str(args.time_id))            
            os.makedirs(args.save_data_dir, exist_ok=True)
            train_df.to_csv(os.path.join(args.save_data_dir, "train.csv"), index=False)
            test_df.to_csv(os.path.join(args.save_data_dir, "test.csv"), index=False)

    return train_df, test_df


def get_test_data(args: Any) -> pd.DataFrame:
    if args.test_text_data_path != "":
        test_df = pd.read_csv(args.test_text_data_path)
        return test_df

    else:    
        if args.test_dataset_path == "":
            seen_dir = os.path.join(args.dataset_dir, "samples", "splits", "seen")
            args.test_dataset_path = os.path.join(seen_dir, "test.csv")

        test_data_df = pd.read_csv(args.test_dataset_path)
            
        metadata_dir = os.path.join(args.dataset_dir, "samples", "metadata")
        users_df = None
        if args.user_description_flag:
            if args.users_path == "":
                args.users_path = os.path.join(metadata_dir, "users.csv")
            users_df = pd.read_csv(args.users_path)

        items_df = None
        if args.item_description_flag:
            if args.items_path == "":
                args.items_path = os.path.join(metadata_dir, "items.csv")
            items_df = pd.read_csv(args.items_path)

        p5_data_creator = P5DataCreator(args)
        test_df = p5_data_creator.create_dataset(
            test_data_df,
            users_df=users_df,
            items_df=items_df
        )

        if args.save_data_flag:
            if args.save_data_dir == "":
                args.save_data_dir = os.path.join(args.dataset_dir, "samples", str(args.time_id))            
            os.makedirs(args.save_data_dir, exist_ok=True)
            test_df.to_csv(os.path.join(args.save_data_dir, "test.csv"), index=False)

    return test_df

def main(args):
    set_seed(args)

    if args.dataset_dir == "":
        args.dataset_dir = os.path.join(args.base_dir, args.dataset_name)

    if args.train_flag:
        train_df, test_df = get_train_test_data(args)
        main_train_test(train_df, test_df, args)
    else:
        test_df = get_test_data(args)
        main_test(test_df, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-base")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="google/flan-t5-base")
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=128)

    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--train_dataset_path", type=str, default="")
    parser.add_argument("--test_dataset_path", type=str, default="")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--users_path", type=str, default="")
    parser.add_argument("--items_path", type=str, default="")
    parser.add_argument("--train_text_data_path", type=str, default="")
    parser.add_argument("--test_text_data_path", type=str, default="")

    parser.add_argument("--train_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(train_flag=True)
    parser.add_argument("--save_data_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(save_data_flag=False)
    parser.add_argument("--save_data_dir", type=str, default="")

    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="")

    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_model_path", type=str, default="")

    parser.add_argument("--prompt_type", type=int, default=0)
    parser.add_argument('--max_description_length', type=int, default=128)
    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)
    parser.add_argument("--user_id_column", type=str, default="user_id")
    parser.add_argument("--item_id_column", type=str, default="item_id")
    parser.add_argument("--rating_column", type=str, default="rating")
    parser.add_argument("--review_column", type=str, default="review")

    parser.add_argument("--user_description_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(user_description_flag=False)
    parser.add_argument("--item_description_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(item_description_flag=True)
    parser.add_argument("--user_description_column", type=str, default="description")
    parser.add_argument("--item_description_column", type=str, default="description")

    parser.add_argument("--truncate_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(truncate_flag=True)
    parser.add_argument("--lower_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(lower_flag=True)
    parser.add_argument("--delete_balise_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_balise_flag=True)
    parser.add_argument("--delete_stopwords_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_stopwords_flag=False)
    parser.add_argument("--delete_punctuation_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_punctuation_flag=False)
    parser.add_argument("--delete_non_ascii_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_non_ascii_flag=True)
    parser.add_argument("--delete_digit_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_digit_flag=False)
    parser.add_argument("--replace_maj_word_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(replace_maj_word_flag=False)
    parser.add_argument("--first_line_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(first_line_flag=False)
    parser.add_argument("--last_line_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(last_line_flag=False)
    parser.add_argument("--stem_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(stem_flag=False)
    parser.add_argument("--lemmatize_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(lemmatize_flag=False)

    args = parser.parse_args()

    if args.prompt_type == 2:
        args.target_rating_flag = False
        args.target_review_flag = True
    else:
        args.target_rating_flag = True
        args.target_review_flag = False

    main(args)
