import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, get_dataset_config_names
import json
from tqdm import tqdm

from src.visualisation.chart import simple_subplot


def parse_args():
    parser = argparse.ArgumentParser(description="Explore the dataset")
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='BEIJING_SUBWAY_30MIN',
        help='Name of the dataset to explore'
    )
    parser.add_argument(
        '--num_series',
        type=int,
        default=5,
        help='Number of series to plot'
    )
    parser.add_argument(
        '--display_names',
        action='store_true',
        help='Whether to display available dataset names'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='The name of the output file to save extracted subsets'
    )
    parser.add_argument(
        '--extract_per_split',
        type=int,
        default=1000,
        help='Number of series to extract per split'
    )

    return parser.parse_args()


def extract_and_save_subsets(
        splits: list[str],
        n_per_config: int,
        output_file: str,
        max_workers: int = 10
):
    def process_split(split_name):
        ds_stream = load_dataset(
            "theforecastingcompany/GiftEvalPretrain",
            name=split_name,
            split='train',
            streaming=True
        ).shuffle().take(n_per_config)

        results = []
        for example in ds_stream:
            results.append(json.dumps(example, default=str, ensure_ascii=False) + "\n")
        return results

    with open(output_file, "w", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_split, split): split for split in splits}

            for future in tqdm(as_completed(futures), total=len(splits)):
                split_results = future.result()
                for line in split_results:
                    fh.write(line)


if __name__ == "__main__":
    args = parse_args()

    splits = get_dataset_config_names("theforecastingcompany/GiftEvalPretrain")

    if args.display_names:
        print(f"Available dataset names ({len(splits)}) : ")

        for name in splits:
            print(f"- {name}")

        ds_name = input("Enter the dataset name to explore: ")
        args.dataset_name = ds_name

    raw_dataset = load_dataset(
        "theforecastingcompany/GiftEvalPretrain",
        name=args.dataset_name,
        split="train",
        streaming=True
    )

    if args.output_file:
        extract_and_save_subsets(
            splits=splits,
            n_per_config=args.extract_per_split,
            output_file=args.output_file
        )

        print(f"Extracted subsets saved to {args.output_file}")

    data_list = list(raw_dataset.take(args.num_series))
    series_list = [item['target'][0] for item in data_list]
    labels = [f"Series {i+1}" for i in range(len(series_list))]

    simple_subplot(series_list, labels)
