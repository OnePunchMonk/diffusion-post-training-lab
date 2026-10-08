import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from dptlab.training.common import epochs_for_optimizer_steps


class _ToyDataset(Dataset):
    """3 items, matching the klein sweep's smallest subject split."""

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.full((4,), float(idx))


def test_epochs_for_optimizer_steps_accounts_for_accumulation():
    # 3 micro-batches, accumulation 4 -> 1 optimizer step per epoch, not 3.
    assert epochs_for_optimizer_steps(num_batches=3, gradient_accumulation_steps=4, max_train_steps=10) == 10
    # No accumulation: one optimizer step per micro-batch, as before.
    assert epochs_for_optimizer_steps(num_batches=3, gradient_accumulation_steps=1, max_train_steps=10) == 4


def test_toy_training_loop_takes_exactly_max_train_steps_optimizer_steps():
    """Regression test for the bug in training/common.py §2.1: a 3-item
    dataset with gradient_accumulation_steps=4 must still deliver exactly
    `max_train_steps` optimizer.step() calls, not `ceil(max_train_steps /
    len(dataloader))` epochs' worth of micro-batches.
    """
    gradient_accumulation_steps = 4
    max_train_steps = 10

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataloader = DataLoader(_ToyDataset(), batch_size=1, shuffle=False)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # `AcceleratedOptimizer.step()` is called on every micro-batch but only
    # does real work when `sync_gradients` is true; count *effective* steps
    # the same way the recipes do.
    effective_step_calls = 0

    global_step = 0
    max_epochs = epochs_for_optimizer_steps(len(dataloader), gradient_accumulation_steps, max_train_steps)

    for _epoch in range(max_epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                loss = model(batch).sum()
                accelerator.backward(loss)
                optimizer.step()
                if accelerator.sync_gradients:
                    effective_step_calls += 1
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step >= max_train_steps:
                    break
        if global_step >= max_train_steps:
            break

    assert effective_step_calls == max_train_steps
    assert global_step == max_train_steps
