import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from typing import Optional, List

from transformers import AutoTokenizer
from datasets import Dataset, load_dataset, get_dataset_config_names
from contaminator.sentence_contaminator import Contaminator


class DatasetContaminator:
    """
    A class to handle dataset contamination and Hub operations.
    """
    
    def __init__(self, tokenizer_path: str, magikarp_path: Optional[str] = None, 
                 glitchminer_path: Optional[str] = None):
        """
        Initialize the DatasetContaminator.
        
        Args:
            tokenizer_path: Path to the tokenizer
            magikarp_path: Path to magikarp output file (JSONL format)
            glitchminer_path: Path to glitchminer output file (pickle format)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.contaminator = Contaminator(
            tokenizer=self.tokenizer,
            magikarp_path=magikarp_path,
            glitchminer_path=glitchminer_path
        )
    
    def load_dataset_subset(self, dataset_name: str, split: str, subset: Optional[str] = None) -> Dataset:
        """
        Load a dataset or specific subset.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub
            split: Dataset split to load
            subset: Optional subset name
            
        Returns:
            Loaded dataset
        """
        if subset:
            return load_dataset(dataset_name, subset, split=split)
        return load_dataset(dataset_name, split=split)
    
    def get_dataset_subsets(self, dataset_name: str) -> List[Optional[str]]:
        """
        Get list of available subsets for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of subset names, or [None] if no subsets exist
        """
        try:
            subsets = get_dataset_config_names(dataset_name)
            print(f"Found subsets: {subsets}")
            return subsets
        except Exception as e:
            print(f"No subsets found: {e}")
            print("Processing main dataset")
            return [None]
    
    def contaminate_dataset(self, dataset_name: str, split: str, prompt_column: str, 
                          subset: Optional[str] = None) -> Dataset:
        """
        Create a contaminated version of the dataset.
        
        Args:
            dataset_name: Name of the dataset
            split: Dataset split to contaminate
            prompt_column: Column name containing text to contaminate
            subset: Optional subset name
            
        Returns:
            Contaminated dataset
        """
        print(f"Loading dataset: {dataset_name}" + (f" (subset: {subset})" if subset else ""))
        dataset = self.load_dataset_subset(dataset_name, split, subset)
        
        def apply_contamination(example):
            """Apply contamination to a single example."""
            try:
                example[prompt_column] = self.contaminator.contaminate_sentence(example[prompt_column])
            except Exception as e:
                print(f"Warning: Failed to contaminate example: {e}")
                
            return example
        
        print(f"Applying contamination to {len(dataset)} examples...")
        contaminated_dataset = dataset.map(apply_contamination)
        print("Contamination completed")
        
        return contaminated_dataset
    
    def push_to_hub(self, dataset: Dataset, output_name: str, split: str, 
                   subset: Optional[str] = None) -> None:
        """
        Push the contaminated dataset to Hugging Face Hub.
        
        Args:
            dataset: Dataset to push
            output_name: Name for the output dataset
            split: Split name
            subset: Optional subset name
        """
        config_name = subset if subset else "default"
        
        print(f"Pushing to Hub: {output_name} (config: {config_name}, split: {split})")
        
        try:
            dataset.push_to_hub(
                output_name,
                config_name=config_name,
                split=split
            )
            print(f"✅ Successfully pushed to {output_name}")
        except Exception as e:
            print(f"❌ Failed to push to Hub: {e}")
            raise
    
    def process_dataset(self, dataset_name: str, output_name: str, split: str, 
                       prompt_column: str) -> None:
        """
        Process entire dataset (all subsets) with contamination and push to Hub.
        
        Args:
            dataset_name: Source dataset name
            output_name: Output dataset name
            split: Dataset split to process
            prompt_column: Column containing text to contaminate
        """
        subsets = self.get_dataset_subsets(dataset_name)
        
        for subset in subsets:
            try:
                print(f"\n{'='*50}")
                print(f"Processing subset: {subset or 'main dataset'}")
                print(f"{'='*50}")
                
                # Contaminate dataset
                contaminated_dataset = self.contaminate_dataset(
                    dataset_name=dataset_name,
                    split=split,
                    prompt_column=prompt_column,
                    subset=subset
                )
                
                self.push_to_hub(
                    dataset=contaminated_dataset,
                    output_name=output_name,
                    split=split,
                    subset=subset
                )
                
            except Exception as e:
                print(f"❌ Error processing subset '{subset}': {e}")
                continue


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create and save contaminated datasets to Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        required=True,
        help="Name of the dataset on Hugging Face Hub"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        required=True,
        help="Path to the tokenizer"
    )
    parser.add_argument(
        "--output_dataset_name", 
        type=str, 
        required=True,
        help="Name for the output contaminated dataset on Hugging Face Hub"
    )

    parser.add_argument(
        "--split", 
        type=str, 
        default="train",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--prompt_col", 
        type=str, 
        default="text",
        help="Column name containing text prompts"
    )
    
    contamination_group = parser.add_mutually_exclusive_group(required=True)
    contamination_group.add_argument(
        "--magikarp_path", 
        type=str,
        help="Path to magikarp output file (JSONL format)"
    )
    contamination_group.add_argument(
        "--glitchminer_path", 
        type=str,
        help="Path to glitchminer output file (pickle format)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        print("Initializing DatasetContaminator...")
        contaminator = DatasetContaminator(
            tokenizer_path=args.tokenizer_path,
            magikarp_path=args.magikarp_path,
            glitchminer_path=args.glitchminer_path
        )
        
        contaminator.process_dataset(
            dataset_name=args.dataset_name,
            output_name=args.output_dataset_name,
            split=args.split,
            prompt_column=args.prompt_col
        )
        
        print("\n🎉 All datasets processed successfully!")
        
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()