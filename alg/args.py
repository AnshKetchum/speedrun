from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass, field
import typing
import os
import torch
os.environ["WANDB_PROJECT"] = "speedrun" 
num_gpus = torch.cuda.device_count()

# main training args; optimizer args + etc needed for overriding trainer defaults + logging
bs = 3
optimizer_name = "cadamw" #"MuonWithAuxAdam"
learning_rate = 5e-3
w_decay = 0.01
lr_aux_adam = 5e-3
decay_aux_adam = 0.01
name = f"qwen3-2b/{optimizer_name}/bs{bs}/{num_gpus}h100s/distil"



def StrBoolTupleType(arg_str: str) -> typing.Tuple[str, bool]:
    if "," in arg_str:
        s, b = arg_str.split(",")
        return str(s), (b.lower() in ("true", "1"))
    else:
        return arg_str, False


@dataclass
class ModelArguments:
    model_name: typing.Optional[str] = field(
        default=None,
        metadata={"help": "model URI or path to finetune."}
    )
    use_lk: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the Liger kernel.",
            "aliases": ["--k"]
        }
    )


@dataclass
class EvalArguments:
    grad_var_stats: bool = True
    binary_grad_similarity_stats: bool = False
    full_grad_similarity_stats: bool = False  # expensive

    harness_benchmarks: typing.List[typing.Dict] = field(
        default_factory= lambda: ["mmlu", "arc_challenge", "arc_easy"],
        # official model release recommendation:
        # include lambda: ["wikitext", "boolq", "hellaswag", "glue", "ai2_arc", "mmlu", "math"]
        metadata={"help": "Benchmarks to compare student and teacher models at end of training."}
    )
    harness_benchmark_limit: int = field(
        default=15000,
        # official model release recommendation: set to None for official releases to measure all data points
        metadata={"help": "Limit the number of examples per task (only use this for testing), If <1, limit is %."}
    )
    harness_benchmark_bootstrap_iters: int = field(
        default=0,
        # official model release recommendation: set to None for official releases to measure error
        metadata={"help": "Number iter for bootstrap stats for stderr. Set to 0 to skip stderr calc. "}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    # optimize convergence to final model
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    lr_scheduler_type: str = "warmup_stable_decay"
    num_decay_steps: int = 2000
    min_lr_ratio: float = 0.1
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr_ratio": 0.1, "num_decay_steps": 2000})
    num_train_epochs: float = 1.0
    max_steps: int = 16000
    slm_distil: bool = False
    distil: bool = True
    distmodel: str = "RedMod/speedrun_model"
    save_only_model: bool = True
    # DDP thing
    ddp_find_unused_parameters: bool = False #True
    # larger batches appear to train better?
    per_device_train_batch_size: int = bs
    gradient_accumulation_steps: int = 1

    # optimize performance and memory
    per_device_eval_batch_size: int = 2  # TODO: auto-find?
    gradient_checkpointing: bool = True
    bf16: bool = True
    torch_compile: bool = True  # TODO: Field

    # Fixes
    gradient_checkpointing_kwargs = {"use_reentrant": False}

    # logging / evaluation
    logging_steps: int = 2
    save_strategy: str = "steps"
    save_steps: int = 4000
    save_total_limit = 16
    eval_strategy: str = "steps"
    eval_steps: int = 2000
    eval_on_start: bool = False
    eval_on_end: bool = False
    report_to: str = "wandb"
    run_name: str = name
    resume: bool = False


parser = HfArgumentParser((
    TrainingArguments,
    ModelArguments,
    EvalArguments
))


def get_args():
    return parser.parse_args_into_dataclasses()
