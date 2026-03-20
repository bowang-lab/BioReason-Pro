import torch
import pytorch_lightning as pl
from gogpt.models.gogpt import GOGPT
from gogpt.config.model_config import GOGPTConfig


class LightningGOGPT(pl.LightningModule):
    def __init__(self, model_args, hparams=None, tokenizer=None):
        super().__init__()

        default_hparams = {
            'learning_rate': 5e-4,
            'weight_decay': 1e-1,
            'beta1': 0.9,
            'beta2': 0.95,
            'warmup_fraction': 0.1,
            'min_lr_ratio': 0.1,
            'log_generations': True,
            'max_logged_generations': 15,
        }

        self.strict_loading = False

        if hparams:
            default_hparams.update(hparams)

        self.save_hyperparameters(default_hparams)

        self.freeze_esm = model_args.get('freeze_esm', True)
        self.esm_learning_rate = model_args.get('esm_learning_rate', 1e-5)

        self.model = GOGPT(GOGPTConfig(**model_args))

        self.tokenizer = tokenizer

        self.tokenizer_info = {
            'mf_start_token_id': model_args['mf_start_token_id'],
            'mf_end_token_id': model_args['mf_end_token_id'],
            'bp_start_token_id': model_args['bp_start_token_id'],
            'bp_end_token_id': model_args['bp_end_token_id'],
            'cc_start_token_id': model_args['cc_start_token_id'],
            'cc_end_token_id': model_args['cc_end_token_id'],
            'pad_token_id': model_args['pad_token_id'],
        }

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        _, loss = self.model(
            protein_tokens=batch["protein_tokens"],
            protein_mask=batch["protein_mask"],
            go_tokens=batch["go_tokens"],
            targets=batch["targets"],
            go_mask=batch["go_mask"],
            organism_id=batch["organism_id"],
        )

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss = self.model(
            protein_tokens=batch["protein_tokens"],
            protein_mask=batch["protein_mask"],
            go_tokens=batch["go_tokens"],
            targets=batch["targets"],
            go_mask=batch["go_mask"],
            organism_id=batch["organism_id"],
        )

        if not hasattr(self, 'validation_step_outputs'):
            self.validation_step_outputs = []

        self.validation_step_outputs.append({
            'loss': loss,
            'batch': batch
        })

        return loss

    def _log_raw_generations_to_wandb(self, raw_generations):
        """Log raw model generations showing complete token sequences with metrics."""
        try:
            raw_data = []

            for sample in raw_generations:
                pred_tokens_str = []
                true_tokens_str = []

                if self.tokenizer:
                    for token_id in sample['pred_tokens']:
                        if token_id in self.tokenizer.id_to_token:
                            pred_tokens_str.append(self.tokenizer.id_to_token[token_id])
                        else:
                            pred_tokens_str.append(f"<UNK:{token_id}>")

                    for token_id in sample['true_tokens']:
                        if token_id in self.tokenizer.id_to_token:
                            true_tokens_str.append(self.tokenizer.id_to_token[token_id])
                        else:
                            true_tokens_str.append(f"<UNK:{token_id}>")

                raw_data.append([
                    sample['aspect'],
                    " ".join(pred_tokens_str),
                    " ".join(true_tokens_str),
                    f"{sample['precision']:.3f}",
                    f"{sample['recall']:.3f}",
                    f"{sample['f1']:.3f}"
                ])

            if raw_data and hasattr(self.logger, 'log_text'):
                columns = ["Aspect", "Raw Predicted Tokens", "Raw True Tokens", "Precision", "Recall", "F1"]
                self.logger.log_text(
                    key="raw_generations",
                    columns=columns,
                    data=raw_data
                )

        except Exception as e:
            print(f"Warning: Could not log raw generations to W&B: {e}")

    def on_validation_epoch_end(self):
        """Compute metrics for all aspects with full batch generation."""
        if not hasattr(self, 'validation_step_outputs') or not self.validation_step_outputs:
            return

        sync_dist = self.trainer.world_size > 1

        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()

        aspect_metrics = {
            'MF': {'f1s': [], 'precisions': [], 'recalls': []},
            'BP': {'f1s': [], 'precisions': [], 'recalls': []},
            'CC': {'f1s': [], 'precisions': [], 'recalls': []}
        }

        raw_generations = []
        MAX_EXAMPLES_TO_LOG = self.hparams.max_logged_generations if self.hparams.log_generations else 0
        examples_logged = 0

        from tqdm import tqdm

        total_examples = sum(output['batch']['go_tokens'].size(0)
                            for output in self.validation_step_outputs)

        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(total=total_examples, desc="Validating", unit="examples")

            for batch_idx, output in enumerate(self.validation_step_outputs):
                batch = output['batch']
                batch_size = batch['go_tokens'].size(0)

                pbar.set_description(f"Validating batch {batch_idx+1} ({batch_size} examples)")

                all_start_tokens = []
                aspects_list = []

                for i in range(batch_size):
                    aspect = self._determine_aspect(batch['go_tokens'][i])
                    aspects_list.append(aspect)

                    if aspect == "MF":
                        start_token = self.tokenizer_info['mf_start_token_id']
                    elif aspect == "BP":
                        start_token = self.tokenizer_info['bp_start_token_id']
                    else:
                        start_token = self.tokenizer_info['cc_start_token_id']
                    all_start_tokens.append(start_token)

                initial_tokens = torch.tensor(all_start_tokens, device=self.device).unsqueeze(1)

                pred_tokens_batch = self.model.generate(
                    protein_tokens=batch['protein_tokens'],
                    protein_mask=batch['protein_mask'],
                    go_tokens=initial_tokens,
                    max_new_tokens=100,
                    temperature=0.5,
                    top_k=20,
                    organism_id=batch['organism_id'],
                )

                for i in range(batch_size):
                    pred_tokens = pred_tokens_batch[i]
                    true_tokens = batch['go_tokens'][i]
                    aspect = aspects_list[i]

                    raw_go_terms = None
                    if 'go_terms_list' in batch and batch['go_terms_list'] is not None:
                        raw_go_terms = batch['go_terms_list'][i]

                    precision, recall, f1 = self._compute_example_metrics(
                        pred_tokens, true_tokens, aspect, raw_go_terms=raw_go_terms
                    )

                    aspect_metrics[aspect]['f1s'].append(f1)
                    aspect_metrics[aspect]['precisions'].append(precision)
                    aspect_metrics[aspect]['recalls'].append(recall)

                    if examples_logged < MAX_EXAMPLES_TO_LOG:
                        raw_generations.append({
                            'aspect': aspect,
                            'true_tokens': true_tokens.tolist()[:50],
                            'pred_tokens': pred_tokens.tolist()[:50],
                            'precision': precision,
                            'recall': recall,
                            'f1': f1
                        })
                        examples_logged += 1

                    pbar.update(1)

            pbar.close()

        aspect_f1s = []
        print("\nValidation Results:")
        print("-" * 50)

        for aspect in ['MF', 'BP', 'CC']:
            if aspect_metrics[aspect]['f1s']:
                n_examples = len(aspect_metrics[aspect]['f1s'])
                aspect_precision = sum(aspect_metrics[aspect]['precisions']) / n_examples
                aspect_recall = sum(aspect_metrics[aspect]['recalls']) / n_examples
                aspect_f1 = sum(aspect_metrics[aspect]['f1s']) / n_examples

                self.log(f'val_{aspect.lower()}_precision', aspect_precision, prog_bar=False, sync_dist=sync_dist)
                self.log(f'val_{aspect.lower()}_recall', aspect_recall, prog_bar=False, sync_dist=sync_dist)
                self.log(f'val_{aspect.lower()}_f1', aspect_f1, prog_bar=False, sync_dist=sync_dist)

                aspect_f1s.append(aspect_f1)

                print(f"{aspect}: P={aspect_precision:.3f}, R={aspect_recall:.3f}, "
                    f"F1={aspect_f1:.3f} (n={n_examples})")

        if self.hparams.log_generations and hasattr(self.logger, 'experiment') and len(raw_generations) > 0:
            self._log_raw_generations_to_wandb(raw_generations)

        self.log('val_loss', avg_loss, prog_bar=True, sync_dist=sync_dist)
        if aspect_f1s:
            overall_f1 = sum(aspect_f1s) / len(aspect_f1s)
            self.log('val_f1', overall_f1, prog_bar=True, sync_dist=sync_dist)
            print(f"\nOverall validation F1: {overall_f1:.3f}")
            print("-" * 50)

        self.validation_step_outputs.clear()
        self.model.train()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log learning rate at each training step."""
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False, prog_bar=False)

        if not self.freeze_esm and len(optimizer.param_groups) > 1:
            esm_lr = optimizer.param_groups[1]['lr']
            self.log('esm_learning_rate', esm_lr, on_step=True, on_epoch=False, prog_bar=False)

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before optimizer step."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self.log('grad_norm', total_norm, on_step=True, on_epoch=False, prog_bar=False)

    def _determine_aspect(self, go_tokens):
        if len(go_tokens) > 0:
            token_id = go_tokens[0].item()
            if token_id == self.tokenizer_info['mf_start_token_id']:
                return "MF"
            elif token_id == self.tokenizer_info['bp_start_token_id']:
                return "BP"
            elif token_id == self.tokenizer_info['cc_start_token_id']:
                return "CC"
        return "MF"

    def _compute_example_metrics(self, pred_tokens, true_tokens, aspect, raw_go_terms=None):
        pred_terms = self._tokens_to_terms(pred_tokens, aspect)

        if raw_go_terms is None:
            raise ValueError(
                "raw_go_terms is required for validation metrics! "
                "Validation data must contain 'go_terms_list' field to ensure metrics are not affected by vocabulary pruning. "
                "If you see this error, there's a bug in data preprocessing."
            )
        true_terms = set(raw_go_terms)

        tp = len(pred_terms & true_terms)
        fp = len(pred_terms - true_terms)
        fn = len(true_terms - pred_terms)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def _tokens_to_terms(self, tokens, aspect):
        """Convert token IDs to GO term strings."""
        token_list = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens

        if aspect == "MF":
            start_token_id = self.tokenizer_info['mf_start_token_id']
            end_token_id = self.tokenizer_info['mf_end_token_id']
        elif aspect == "BP":
            start_token_id = self.tokenizer_info['bp_start_token_id']
            end_token_id = self.tokenizer_info['bp_end_token_id']
        elif aspect == "CC":
            start_token_id = self.tokenizer_info['cc_start_token_id']
            end_token_id = self.tokenizer_info['cc_end_token_id']
        else:
            start_token_id = self.tokenizer_info['mf_start_token_id']
            end_token_id = self.tokenizer_info['mf_end_token_id']

        start_pos = token_list.index(start_token_id) if start_token_id in token_list else -1
        end_pos = token_list.index(end_token_id) if end_token_id in token_list else -1

        if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
            token_list = token_list[start_pos+1:end_pos]
        elif start_pos != -1:
            token_list = token_list[start_pos+1:]

        special_tokens = [
            self.tokenizer_info['pad_token_id'],
            self.tokenizer_info['mf_start_token_id'],
            self.tokenizer_info['mf_end_token_id'],
            self.tokenizer_info['bp_start_token_id'],
            self.tokenizer_info['bp_end_token_id'],
            self.tokenizer_info['cc_start_token_id'],
            self.tokenizer_info['cc_end_token_id']
        ]

        go_terms = set()
        if self.tokenizer is not None:
            for token_id in token_list:
                if token_id not in special_tokens and token_id in self.tokenizer.id_to_token:
                    term = self.tokenizer.id_to_token[token_id]
                    if term.startswith("GO:"):
                        go_terms.add(term)
        else:
            go_terms = {t for t in token_list if t not in special_tokens}

        return go_terms

    def on_save_checkpoint(self, checkpoint):
        if self.freeze_esm:
            keys_to_remove = [k for k in checkpoint['state_dict'].keys()
                              if 'esm' in k or 'protein_model' in k]
            for key in keys_to_remove:
                del checkpoint['state_dict'][key]
            print(f"Removed {len(keys_to_remove)} ESM parameters from checkpoint (frozen mode)")
        else:
            esm_keys = [k for k in checkpoint['state_dict'].keys() if 'esm' in k]
            print(f"Keeping {len(esm_keys)} ESM parameters in checkpoint (finetuning mode)")

    def on_load_checkpoint(self, checkpoint):
        has_esm = any('esm' in k for k in checkpoint['state_dict'].keys())

        if not has_esm and not self.freeze_esm:
            print("Loading frozen checkpoint into finetuning mode - ESM from pretrained model")
        elif has_esm:
            print("Loading finetuning checkpoint with ESM weights")
        else:
            print("Loading frozen checkpoint - ESM from pretrained model")

    def configure_optimizers(self):
        import math
        from torch.optim.lr_scheduler import LambdaLR

        if not self.freeze_esm:
            esm_params = []
            non_esm_params = []

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if 'esm' in name:
                        esm_params.append(param)
                    else:
                        non_esm_params.append(param)

            param_groups = [
                {'params': non_esm_params, 'lr': self.hparams.learning_rate},
                {'params': esm_params, 'lr': self.esm_learning_rate}
            ]

            print(f"Optimizer setup: {len(non_esm_params)} non-ESM params (LR={self.hparams.learning_rate}), "
                  f"{len(esm_params)} ESM params (LR={self.esm_learning_rate})")

            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.hparams.weight_decay,
                betas=(self.hparams.beta1, self.hparams.beta2)
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(self.hparams.beta1, self.hparams.beta2)
            )

        num_warmup_steps = int(self.trainer.estimated_stepping_batches * self.hparams.warmup_fraction)
        num_training_steps = self.trainer.estimated_stepping_batches

        min_lr_ratio = self.hparams.min_lr_ratio

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
