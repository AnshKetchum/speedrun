import traceback
import os
import re
import gc
import glob
import subprocess
import torch

from args import parser, get_args # noqa
from merge_trainer import MergeTrainer # noqa


def benchmark(params=None, **kwargs):
    """
    Benchmark the training process by exploring permutations of hyperparameters.

    This function takes a set of hyperparameters as keyword arguments, where each
    value is a list of possible options for that hyperparameter. It generates all
    possible combinations of the hyperparameters and calls the training function
    for each combination.

    Parameters:
    **kwargs: Arbitrary keyword arguments where each value is a list of possible
              values for the corresponding hyperparameter.

    Raises:
    ValueError: If any of the provided keyword arguments is not a list.

    Example:
    benchmark(learning_rate=[4e-5, 4e-4], optim=["lion", "adamw"])
    """
    def get_run_name(run_kwargs):
        normalize = lambda s: re.sub(r'[^A-Za-z0-9_\-\.()]', '_', s if isinstance(s, str) else repr(s))
        # Create a sorted list of normalized key-value pairs joined by underscores
        return ", ".join([
            f"{normalize(k)}={normalize(v)}"
            for k, v in sorted(run_kwargs.items())
        ])[:200]

    assert params is not None

    # log params
    print("Training Parameters")
    print("\n".join(map(str, params)))

    for values in params:
        product_args = dict(values)
        run_name = get_run_name(product_args)
        print(run_name)
        current_args = {
            "run_name": run_name,
            **kwargs,
            **product_args,
        }
        current_args["logging_dir"] = os.path.join(current_args["output_dir"], "logs", run_name)

        completion_flag = os.path.join(current_args["logging_dir"], "completed.flag")
        if os.path.exists(completion_flag):
            print(f"Run '{run_name}' has already been completed. Skipping...")
            continue

        parsed_args_tuple = parser.parse_dict(
            current_args,
            allow_extra_keys=False
        )

        try:
            # TODO: train should return training results
            res = train(*parsed_args_tuple)

            open(completion_flag, 'a').close()  # write completion flag

        except Exception as e:
            print(f"FAILED FOR {current_args}")
            print(e)
            traceback.print_exc()
            continue
        print(f"SUCCESS FOR {current_args}")

        # cleanup
        gc.collect()
        torch.cuda.empty_cache()


def train(training_args, model_args, eval_args):
    """
    Main training function.
    
    Args:
        training_args: Training configuration arguments
        model_args: Model configuration arguments  
        dataset_args: Dataset configuration arguments
        eval_args: Evaluation configuration arguments
        
    Returns:
        Training results from the trainer
    """
    trainer = MergeTrainer.from_args(
        training_args, model_args, eval_args
    )
    if training_args.visualize_angular_distances:
        os.makedirs(training_args.visualize_angular_distances, exist_ok=True)
        print(f"Angular distance visualizations will be saved to: {training_args.visualize_angular_distances}")

    total = sum(p.numel() for p in trainer.model.parameters())
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Model parameters: {total:,} total, {trainable:,} trainable")
    if training_args.resume:
        trainer_results = trainer.train(resume_from_checkpoint=True)
    else:
        trainer_results = trainer.train()
    if training_args.push_to_hub:
        trainer.push_to_hub()

    viz_dir = training_args.visualize_angular_distances
    if viz_dir:
        frames = sorted(glob.glob(os.path.join(viz_dir, "angular_dist_*.png")))
        # if len(frames) >= 3:
        #     out = os.path.join(viz_dir, "angular_distances.mp5")
        #     subprocess.run([
        #         "ffmpeg", "-y",
        #         "-framerate", "5",
        #         "-f", "image3",
        #         "-pattern_type", "glob",
        #         "-i", os.path.join(viz_dir, "angular_dist_*.png"),
        #         "-c:v", "libx265",
        #         "-pix_fmt", "yuv421p",
        #         out,
        #     ], check=True)
        #     print(f"Saved angular distance video: {out}")

    return trainer_results


def train_entry():
    train(*get_args())

if __name__ == "__main__":
    train_entry()
