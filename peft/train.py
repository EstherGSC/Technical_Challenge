import os
import json
import math
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)


# ============================================================
# 1. 基本配置
# ============================================================

BASE_MODEL = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B-SelfPlay/step-15"

OUTPUT_DIR = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B-PEFT"

DATA_DIR = "/root/autodl-tmp/datasets/physics_peft"

DATASET_NAME = "camel-ai/physics"

SEED = 42

# 数据划分
TRAIN_SIZE = 18000
TEST_SIZE = 2000

# LoRA
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# 只对 FFN 做 LoRA
TARGET_MODULES = {
    "gate_proj",
    "up_proj",
    "down_proj",
}

# 训练
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

MAX_LENGTH = 1536

WARMUP_RATIO = 0.03

# 日志 / 保存
LOG_INTERVAL = 20
SAVE_EVERY_EPOCH = True


# ============================================================
# 2. 随机种子
# ============================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. 手写 LoRA Linear
# ============================================================

class LoRALinear(nn.Module):
    """
    手写 LoRA：

        W' = W + alpha/r * A @ B

    其中：
        W : frozen original weight
        A : trainable
        B : trainable

    按 taskbook 的初始化方式：
        A = 0
        B = Gaussian random
    """

    def __init__(
        self,
        original_linear,
        r=8,
        alpha=32,
        dropout=0.05,
    ):
        super().__init__()

        if not isinstance(original_linear, nn.Linear):
            raise TypeError(
                f"LoRALinear only supports nn.Linear, "
                f"got {type(original_linear)}"
            )

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # ----------------------------------------------------
        # 保存原始 Linear
        # ----------------------------------------------------
        #
        # 不复制 weight，直接持有原始模块。
        # 这样可以节省一份 1.5B 参数的额外内存。
        #
        self.original_linear = original_linear

        # 冻结原始参数
        self.original_linear.weight.requires_grad = False

        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        # ----------------------------------------------------
        # LoRA 参数
        # ----------------------------------------------------

        # A: [out_features, r]
        #
        # 按 taskbook：
        # A 初始化为 0
        #
        self.lora_A = nn.Parameter(
            torch.zeros(
                self.out_features,
                r,
                dtype=original_linear.weight.dtype,
                device=original_linear.weight.device,
            )
        )

        # B: [r, in_features]
        #
        # B Gaussian 初始化
        #
        self.lora_B = nn.Parameter(
            torch.empty(
                r,
                self.in_features,
                dtype=original_linear.weight.dtype,
                device=original_linear.weight.device,
            )
        )

        nn.init.normal_(
            self.lora_B,
            mean=0.0,
            std=0.02,
        )

        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x):

        # 原始 Linear
        result = self.original_linear(x)

        # LoRA branch
        lora_x = self.lora_dropout(x)

        # x @ B^T
        lora_x = torch.matmul(
            lora_x,
            self.lora_B.t(),
        )

        # (...) @ A^T
        lora_x = torch.matmul(
            lora_x,
            self.lora_A.t(),
        )

        result = result + self.scaling * lora_x

        return result


# ============================================================
# 4. 替换目标 Linear
# ============================================================

def replace_lora_modules(
    model,
    target_modules,
    r,
    alpha,
    dropout,
):
    """
    将模型中的：

        gate_proj
        up_proj
        down_proj

    替换成：

        LoRALinear
    """

    replaced = []

    def recursive_replace(module, prefix=""):

        for name, child in list(module.named_children()):

            full_name = (
                f"{prefix}.{name}"
                if prefix
                else name
            )

            # ------------------------------------------------
            # 当前 child 是否是目标 Linear
            # ------------------------------------------------

            if (
                name in target_modules
                and isinstance(child, nn.Linear)
            ):

                lora_layer = LoRALinear(
                    original_linear=child,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )

                setattr(
                    module,
                    name,
                    lora_layer,
                )

                replaced.append(full_name)

            else:
                recursive_replace(
                    child,
                    full_name,
                )

    recursive_replace(model)

    return replaced


# ============================================================
# 5. 冻结模型，只训练 LoRA
# ============================================================

def freeze_non_lora_parameters(model):

    trainable = 0
    total = 0

    for name, param in model.named_parameters():

        total += param.numel()

        if (
            "lora_A" in name
            or "lora_B" in name
        ):
            param.requires_grad = True
            trainable += param.numel()

        else:
            param.requires_grad = False

    return trainable, total


# ============================================================
# 6. Dataset
# ============================================================

class PhysicsDataset(Dataset):

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        max_length=1536,
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]

        question = item["message_1"]
        solution = item["message_2"]

        # ----------------------------------------------------
        # 使用一个清晰的 physics instruction 格式
        # ----------------------------------------------------

        prompt = (
            "Solve the following physics problem. "
            "Provide a detailed step-by-step solution.\n\n"
            "Problem:\n"
            + question
            + "\n\n"
            "Solution:\n"
        )

        full_text = prompt + solution

        # ----------------------------------------------------
        # 分别 tokenize prompt 和完整文本
        # ----------------------------------------------------

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        # ----------------------------------------------------
        # prompt 已经超过 max_length
        # 这种样本无法留下有效 solution token
        # ----------------------------------------------------

        if len(prompt_ids) >= self.max_length:

            # 至少保证不会报错
            input_ids = full_ids

            labels = [-100] * len(input_ids)

        else:

            input_ids = full_ids

            prompt_len = min(
                len(prompt_ids),
                len(input_ids),
            )

            labels = (
                [-100] * prompt_len
                + input_ids[prompt_len:]
            )

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================
# 7. Collator
# ============================================================

class PhysicsCollator:

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

        if tokenizer.pad_token_id is None:

            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, batch):

        max_len = max(
            len(x["input_ids"])
            for x in batch
        )

        input_ids = []
        attention_mask = []
        labels = []

        pad_id = self.tokenizer.pad_token_id

        for item in batch:

            length = len(item["input_ids"])
            pad_len = max_len - length

            input_ids.append(
                item["input_ids"]
                + [pad_id] * pad_len
            )

            attention_mask.append(
                item["attention_mask"]
                + [0] * pad_len
            )

            labels.append(
                item["labels"]
                + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


# ============================================================
# 8. 数据集划分
# ============================================================

def prepare_dataset():

    os.makedirs(
        DATA_DIR,
        exist_ok=True,
    )

    train_path = os.path.join(
        DATA_DIR,
        "train_18000",
    )

    test_path = os.path.join(
        DATA_DIR,
        "test_2000",
    )

    # --------------------------------------------------------
    # 如果之前已经保存过划分，直接读取
    # --------------------------------------------------------

    if (
        os.path.exists(train_path)
        and os.path.exists(test_path)
    ):

        print("=" * 80)
        print("Loading existing physics split...")
        print("=" * 80)

        train_dataset = HFDataset.load_from_disk(
            train_path
        )

        test_dataset = HFDataset.load_from_disk(
            test_path
        )

        print(
            f"Train: {len(train_dataset)}"
        )

        print(
            f"Test : {len(test_dataset)}"
        )

        return train_dataset, test_dataset

    # --------------------------------------------------------
    # 第一次运行：下载完整 20K
    # --------------------------------------------------------

    print("=" * 80)
    print("Loading camel-ai/physics...")
    print("=" * 80)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    print(
        f"Original dataset size: {len(dataset)}"
    )

    assert len(dataset) >= 20000, (
        f"Expected at least 20000 examples, "
        f"got {len(dataset)}"
    )

    # --------------------------------------------------------
    # 固定 seed
    # --------------------------------------------------------

    split = dataset.train_test_split(
        test_size=TEST_SIZE,
        seed=SEED,
    )

    train_dataset = split["train"]
    test_dataset = split["test"]

    # --------------------------------------------------------
    # 如果原始数据超过 20K，只取 18K + 2K
    # --------------------------------------------------------

    train_dataset = train_dataset.select(
        range(TRAIN_SIZE)
    )

    test_dataset = test_dataset.select(
        range(TEST_SIZE)
    )

    print("=" * 80)
    print("Dataset split")
    print("=" * 80)

    print(
        f"Train: {len(train_dataset)}"
    )

    print(
        f"Test : {len(test_dataset)}"
    )

    assert len(train_dataset) == 18000
    assert len(test_dataset) == 2000

    # --------------------------------------------------------
    # 保存固定划分
    # --------------------------------------------------------

    train_dataset.save_to_disk(
        train_path
    )

    test_dataset.save_to_disk(
        test_path
    )

    print()
    print(
        f"Saved train split -> {train_path}"
    )

    print(
        f"Saved test split  -> {test_path}"
    )

    return train_dataset, test_dataset


# ============================================================
# 9. 保存 LoRA Adapter
# ============================================================

def save_lora_adapter(
    model,
    tokenizer,
    output_dir,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    lora_state = {}

    for name, param in model.named_parameters():

        if (
            "lora_A" in name
            or "lora_B" in name
        ):
            lora_state[name] = (
                param.detach()
                .cpu()
                .clone()
            )

    adapter_path = os.path.join(
        output_dir,
        "adapter_model.pt",
    )

    torch.save(
        lora_state,
        adapter_path,
    )

    config = {
        "base_model": BASE_MODEL,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": sorted(
            list(TARGET_MODULES)
        ),
    }

    config_path = os.path.join(
        output_dir,
        "adapter_config.json",
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    tokenizer.save_pretrained(
        output_dir
    )

    print()
    print("=" * 80)
    print("LoRA adapter saved")
    print("=" * 80)

    print(
        f"Adapter: {adapter_path}"
    )

    print(
        f"Config : {config_path}"
    )


# ============================================================
# 10. 主训练
# ============================================================

def main():

    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device("cuda")

    print("=" * 80)
    print("PEFT / Manual LoRA Training")
    print("=" * 80)

    print(
        f"Base model : {BASE_MODEL}"
    )

    print(
        f"Output     : {OUTPUT_DIR}"
    )

    print(
        f"Dataset    : {DATASET_NAME}"
    )

    print(
        f"Train      : {TRAIN_SIZE}"
    )

    print(
        f"Test       : {TEST_SIZE}"
    )

    print(
        f"LoRA       : r={LORA_R}, "
        f"alpha={LORA_ALPHA}, "
        f"dropout={LORA_DROPOUT}"
    )

    print(
        f"Target     : "
        f"{sorted(TARGET_MODULES)}"
    )

    print(
        f"Epochs     : {NUM_EPOCHS}"
    )

    print(
        f"LR         : {LEARNING_RATE}"
    )

    print(
        f"Batch      : {BATCH_SIZE}"
    )

    print(
        f"Grad accum : "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Max length: {MAX_LENGTH}"
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_raw, test_raw = prepare_dataset()

    print()
    print(
        "Building training dataset..."
    )

    train_dataset = PhysicsDataset(
        train_raw,
        tokenizer,
        max_length=MAX_LENGTH,
    )

    # test_dataset 这里只保存，不参与训练
    #
    # 后面评估时使用：
    #
    # /root/autodl-tmp/datasets/physics_peft/test_2000
    #
    test_dataset = test_raw

    print(
        f"Training examples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Held-out test examples: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    collator = PhysicsCollator(
        tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Loading base model...")
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # --------------------------------------------------------
    # Gradient checkpointing
    # --------------------------------------------------------

    if hasattr(
        model,
        "gradient_checkpointing_enable",
    ):

        model.gradient_checkpointing_enable()

    if hasattr(
        model,
        "enable_input_require_grads",
    ):

        model.enable_input_require_grads()

    # --------------------------------------------------------
    # 插入 LoRA
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Replacing target modules with LoRA...")
    print("=" * 80)

    replaced = replace_lora_modules(
        model=model,
        target_modules=TARGET_MODULES,
        r=LORA_R,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
    )

    print(
        f"Replaced {len(replaced)} modules."
    )

    for name in replaced:
        print(
            f"  [LoRA] {name}"
        )

    # --------------------------------------------------------
    # 冻结 base，只训练 LoRA
    # --------------------------------------------------------

    trainable_params, total_params = (
        freeze_non_lora_parameters(
            model
        )
    )

    trainable_ratio = (
        trainable_params / total_params
    )

    print()
    print("=" * 80)
    print("Parameter statistics")
    print("=" * 80)

    print(
        f"Total parameters     : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_params:,}"
    )

    print(
        f"Trainable ratio      : "
        f"{trainable_ratio:.6%}"
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    model.to(device)

    model.train()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    total_update_steps = math.ceil(
        len(train_loader)
        / GRADIENT_ACCUMULATION_STEPS
    ) * NUM_EPOCHS

    warmup_steps = int(
        total_update_steps
        * WARMUP_RATIO
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    print()
    print(
        f"Optimizer steps: "
        f"{total_update_steps}"
    )

    print(
        f"Warmup steps   : "
        f"{warmup_steps}"
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    global_step = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    for epoch in range(NUM_EPOCHS):

        print()
        print("=" * 80)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )
        print("=" * 80)

        running_loss = 0.0
        update_count = 0

        for batch_idx, batch in enumerate(
            train_loader
        ):

            input_ids = batch[
                "input_ids"
            ].to(
                device,
                non_blocking=True,
            )

            attention_mask = batch[
                "attention_mask"
            ].to(
                device,
                non_blocking=True,
            )

            labels = batch[
                "labels"
            ].to(
                device,
                non_blocking=True,
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )

                loss = outputs.loss

            loss_for_backward = (
                loss
                / GRADIENT_ACCUMULATION_STEPS
            )

            loss_for_backward.backward()

            running_loss += loss.item()

            # ------------------------------------------------
            # Gradient accumulation
            # ------------------------------------------------

            if (
                (batch_idx + 1)
                % GRADIENT_ACCUMULATION_STEPS
                == 0
            ):

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [
                        p
                        for p in model.parameters()
                        if p.requires_grad
                    ],
                    max_norm=1.0,
                )

                optimizer.step()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1
                update_count += 1

                if (
                    global_step
                    % LOG_INTERVAL
                    == 0
                ):

                    avg_loss = (
                        running_loss
                        / LOG_INTERVAL
                    )

                    lr = scheduler.get_last_lr()[0]

                    print(
                        f"epoch={epoch + 1} "
                        f"step={global_step} "
                        f"batch={batch_idx + 1}/"
                        f"{len(train_loader)} "
                        f"loss={avg_loss:.6f} "
                        f"lr={lr:.3e}"
                    )

                    running_loss = 0.0

        # ----------------------------------------------------
        # Epoch end
        # ----------------------------------------------------

        if update_count > 0:

            epoch_avg_loss = (
                running_loss / update_count
            )

        else:

            epoch_avg_loss = 0.0

        print()
        print(
            f"Epoch {epoch + 1} finished."
        )

        print(
            f"Global step: {global_step}"
        )

        print(
            f"Current LR: "
            f"{scheduler.get_last_lr()[0]:.6e}"
        )

        # ----------------------------------------------------
        # 保存 checkpoint
        # ----------------------------------------------------

        if SAVE_EVERY_EPOCH:

            epoch_dir = os.path.join(
                OUTPUT_DIR,
                f"epoch-{epoch + 1}",
            )

            save_lora_adapter(
                model=model,
                tokenizer=tokenizer,
                output_dir=epoch_dir,
            )

    # ========================================================
    # 最终 adapter
    # ========================================================

    final_dir = os.path.join(
        OUTPUT_DIR,
        "final",
    )

    save_lora_adapter(
        model=model,
        tokenizer=tokenizer,
        output_dir=final_dir,
    )

    # --------------------------------------------------------
    # 保存训练信息
    # --------------------------------------------------------

    training_info = {
        "base_model": BASE_MODEL,
        "dataset": DATASET_NAME,
        "train_size": TRAIN_SIZE,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,
        "max_length": MAX_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules":
            sorted(list(TARGET_MODULES)),
        "trainable_parameters":
            trainable_params,
        "total_parameters":
            total_params,
        "trainable_ratio":
            trainable_ratio,
    }

    info_path = os.path.join(
        OUTPUT_DIR,
        "training_info.json",
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    with open(
        info_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            training_info,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("PEFT TRAINING FINISHED")
    print("=" * 80)

    print(
        f"Final adapter: {final_dir}"
    )

    print(
        f"Physics test set: "
        f"{DATA_DIR}/test_2000"
    )

    print(
        "The 2000 test examples were NOT "
        "used during training."
    )


# ============================================================
# 11. Entry
# ============================================================

if __name__ == "__main__":
    main()

