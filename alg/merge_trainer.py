from dataclasses import asdict
import statistics
import collections
import os
import gc
import json

import transformers
import torch
from huggingface_hub import ModelCard
from args import *
from args import optimizer_name, w_decay, learning_rate, lr_aux_adam, decay_aux_adam
from data import *
from metrics import *
from models import *
import objectives
import datasets
from torch.optim.lr_scheduler import LambdaLR
from cadamw import AdamW
from muon import MuonWithAuxAdam
from transformers.training_args import OptimizerNames
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

def zero_grad(grad):
    return grad.zero_()

class MergeTrainer(transformers.Trainer):
    def __init__(
        self,
        objective,
        model,
        tokenizer,
        evaluators,
        all_args=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, model=model, tokenizer=tokenizer, **kwargs)

        self.all_args = all_args or {}

        self.objective = objective

        self.evaluators = evaluators

        self.muon = False

        self._prev_grad = None
        self._prev_grad_8bit = None
        self._prev_grad_4bit = None
        self._prev_grad_sign = None
        self._extra_stats = []
        self._rolling_grad_norms = collections.deque(maxlen=16)

    @classmethod
    def from_args(
            cls,
            training_args,
            model_args,
            eval_args
    ):
        model_kwargs = {}
        if training_args.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif training_args.fp16:
            model_kwargs["torch_dtype"] = torch.float16

        model, tokenizer = get_model_tokenizer(model_args, **model_kwargs)
        print("got model", model)
        print("got tokenizer", tokenizer)
        print("starting datasets")
        train_dataset = get_dataset()
        # this is a dummy for trainer
        test_dataset = get_dataset()
        objective = objectives.Objective(distil =training_args.distil, slm_distil= training_args.slm_distil, name= training_args.distmodel)

        return cls(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            objective=objective,
            evaluators=None,
            all_args=dict(
                model_args=model_args,
                eval_args=eval_args,
            )
        )

    @classmethod
    def from_kwargs(cls, **kwargs):
        parsed_args_tuple = parser.parse_dict(
            kwargs,
            allow_extra_keys=False
        )
        return cls.from_args(*parsed_args_tuple)
    
    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args, model):
        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAMW_HF:
            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        else:
            optimizer_cls, optimizer_kwargs = super().get_optimizer_cls_and_kwargs(args, model)
        return optimizer_cls, optimizer_kwargs

    def train(self, *args, **kwargs):
        train_output = super().train(*args, **kwargs)
        if self.args.eval_on_end:
            self.evaluate()
        bench_metrics_out = self._maybe_benchmark()
        if bench_metrics_out is None:
            return train_output
        return transformers.trainer_utils.TrainOutput(
            train_output.global_step,
            train_output.training_loss,
            {**train_output.metrics, **bench_metrics_out}
        )
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model
        fw_params = [p for p in opt_model.model.layers.parameters() if p.ndim >= 2]
        non_proj = [p for p in opt_model.model.layers.parameters() if p.ndim < 2]
        non_proj.extend(opt_model.lm_head.parameters())
        non_proj.extend(opt_model.model.embed_tokens.parameters())
        
        if optimizer_name == "MuonWithAuxAdam":
            optimizer = MuonWithAuxAdam([
                    dict(params=fw_params, use_muon=True,
                            lr=learning_rate, weight_decay=w_decay),
                    dict(params=non_proj, use_muon=False,
                            lr=lr_aux_adam, betas=(0.99, 0.999), weight_decay=decay_aux_adam),
                ])
            self.muon = True
        elif optimizer_name == "cadamw":
            optimizer = AdamW([
                    {'params': fw_params},
                    {'params': non_proj}
                ], lr=learning_rate, weight_decay=w_decay)
        else:
            raise ValueError
        
        self.optimizer = optimizer
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, return_stats=False):
        del inputs["labels"]

        loss_dict = self.objective.forward(model, inputs)
        loss = loss_dict.pop("loss")

        stats = {k: float(v) for k, v in loss_dict.items()}

        if return_outputs:
            # TODO: real output, this is nothing of use
            return loss, torch.tensor([1.0])
        elif stats:
            return loss, stats
        else:
            return loss

    def training_step(self, model, inputs, num_items_in_batch = None) -> torch.Tensor:
        """
        Copy of https://github.com/huggingface/transformers/blob/52a021375/src/transformers/trainer.py#L3394

        With additional behavior:
        - Enable comparative gradient logging
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        if transformers.trainer.is_sagemaker_mp_enabled():
            loss_mb = transformers.trainer.smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, stats = self.compute_loss(model, inputs, return_stats=True)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if transformers.trainer.is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif transformers.trainer.is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif transformers.trainer.is_torch_musa_available():
                torch.musa.empty_cache()
            elif transformers.trainer.is_torch_npu_available():
                torch.npu.empty_cache()
            elif transformers.trainer.is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [transformers.trainer.OptimizerNames.LOMO, transformers.trainer.OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with transformers.trainer.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        self._extra_stats.append(stats)

        ##############
        # END NEW CODE
        ##############
        self.objective.count = self.state.global_step

        return loss.detach() / self.args.gradient_accumulation_steps

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time=None, learning_rate=None):
        """
        Copy of https://github.com/huggingface/transformers/blob/52a021375/src/transformers/trainer.py#L3394

        With additional behavior:
        - Enable gradient variance logging
        - clear self._extra_stats once queued for logging
        """
        
        self._rolling_grad_norms.append(grad_norm.item())
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if transformers.trainer.is_torch_xla_available():
                transformers.trainer.xm.mark_step()

            logs = {}

            ##############
            # NEW CODE
            ##############

            transposed_stats = collections.defaultdict(list)
            [transposed_stats[key].append(d.get(key)) for d in self._extra_stats for key in d]
            for k in transposed_stats:
                if k[0] != "_":
                    logs[k] = sum(transposed_stats[k]) / len(transposed_stats[k])

            if len(self._rolling_grad_norms) == 16 and self.all_args["eval_args"].grad_var_stats:
                logs["grad_norm_var"] = statistics.variance(self._rolling_grad_norms)

            self._extra_stats = []

            ##############
            # END NEW CODE
            ##############

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        if self.control.should_save:
            #import copy
            # for_save_model = copy.deepcopy(model).to("cpu")
            # merge the weights in
            # Merge the weights in modules with 'attn' or 'mlp' in the name and the right attribute
            #for name, module in for_save_model.named_modules():
            #    if ("attn" in name or "mlp" in name) and hasattr(module, "merge_weights"):
            #        module.merge_weights()
            #self._save_checkpoint(for_save_model, trial)
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
        metrics = None
        
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

    def evaluate(self, *args, metric_key_prefix="eval", **kwargs):
        self.model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        metrics = {}
        results = (self.bench_single())
        print(results)
        self.model.train()
        gc.collect()
        torch.cuda.empty_cache()
        return metrics

    def _maybe_benchmark(self):
        if not self.all_args.get("eval_args") or not self.all_args["eval_args"].harness_benchmarks:
            return

        benchmarks = self.all_args["eval_args"].harness_benchmarks
        limit = self.all_args["eval_args"].harness_benchmark_limit
        bootstrap_iters = self.all_args["eval_args"].harness_benchmark_bootstrap_iters

        self.model.eval()
        gc.collect()
        torch.cuda.empty_cache()

        # Load existing benchmarks if file exists
        benchmarks_file = self.benchmarks_json
        db = {}
        if os.path.exists(benchmarks_file):
            try:
                with open(benchmarks_file, 'r') as f:
                    db = json.load(f)
            except (json.JSONDecodeError, IOError):
                db = {}
        
        # Run student benchmarks only
        student_metrics = run_benchmarks(
            self.model, self.tokenizer, benchmarks, limit, bootstrap_iters
        )
        db[self.args.run_name] = self._serialize_metrics(student_metrics)
        
        # Save to JSON (only from rank 0 to avoid conflicts)
        local_rank = getattr(self.args, 'local_rank', -1)
        if local_rank in [-1, 0]:
            with open(benchmarks_file, 'w') as f:
                json.dump(db, f, indent=2)
        
        print(student_metrics)

        return student_metrics
    
    def bench_single(self):
        if not self.all_args.get("eval_args") or not self.all_args["eval_args"].harness_benchmarks:
            return

        benchmarks = self.all_args["eval_args"].harness_benchmarks
        limit = self.all_args["eval_args"].harness_benchmark_limit
        bootstrap_iters = self.all_args["eval_args"].harness_benchmark_bootstrap_iters

        self.model.eval()
        gc.collect()
        torch.cuda.empty_cache()

        # Load existing benchmarks if file exists
        benchmarks_file = self.benchmarks_json
        db = {}
        if os.path.exists(benchmarks_file):
            try:
                with open(benchmarks_file, 'r') as f:
                    db = json.load(f)
            except (json.JSONDecodeError, IOError):
                db = {}
        
        # Run student benchmarks
        student_metrics = run_benchmarks(
            self.model, self.tokenizer, benchmarks, limit, bootstrap_iters
        )
        # Include step number in key to preserve evaluations at different steps
        eval_key = f"{self.args.run_name}_step{self.state.global_step}"
        db[eval_key] = self._serialize_metrics(student_metrics)
        
        # Save to JSON (only from rank 0 to avoid conflicts)
        local_rank = getattr(self.args, 'local_rank', -1)
        if local_rank in [-1, 0]:
            with open(benchmarks_file, 'w') as f:
                json.dump(db, f, indent=2)
        self.model.train()
        return student_metrics

    @property
    def benchmarks_json(self):
        return os.path.join(self.args.output_dir, "benchmarks.json")
    
    def _serialize_metrics(self, metrics):
        """Convert metrics to JSON-serializable format, keeping only summary results."""
        if metrics is None:
            return None
        
        # Extract only the results/summary, skip task configurations
        if isinstance(metrics, dict):
            # If there's a "results" key, use that (contains summary metrics)
            if "results" in metrics:
                return self._serialize_metrics(metrics["results"])
            
            # Otherwise, filter out configuration keys and keep only metric values
            serialized = {}
            skip_keys = {
                "description", "target_delimiter", "fewshot_delimiter", 
                "fewshot_config", "num_fewshot", "metric_list", "config",
                "task_name", "task_info", "dataset_path", "output_type"
            }
            
            for key, value in metrics.items():
                # Skip configuration keys
                if key in skip_keys:
                    continue
                
                # Keep metric values (typically numeric)
                if isinstance(value, dict):
                    # Recursively process nested dicts
                    nested = self._serialize_metrics(value)
                    if nested:  # Only add if not empty
                        serialized[key] = nested
                elif isinstance(value, (int, float, str, bool, type(None))):
                    serialized[key] = value
                elif hasattr(value, 'item'):  # torch.Tensor or numpy scalar
                    serialized[key] = float(value.item())
                elif isinstance(value, (list, tuple)):
                    # Only keep lists of simple values, not config lists
                    if len(value) > 0 and not isinstance(value[0], dict):
                        serialized[key] = [
                            float(v.item()) if hasattr(v, 'item') else v 
                            for v in value
                        ]
            return serialized
        
        # For non-dict values, convert to serializable format
        if isinstance(metrics, (int, float, str, bool, type(None))):
            return metrics
        elif hasattr(metrics, 'item'):
            return float(metrics.item())
        else:
            return str(metrics)
