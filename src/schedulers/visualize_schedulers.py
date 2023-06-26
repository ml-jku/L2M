import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.optim.lr_scheduler import SequentialLR
from lr_schedulers import make_lr_scheduler


def visualize_lr_scheduler(lr_scheduler, total_steps=1e6, title="", save_dir=None):
    total_steps = int(total_steps)
    lrs = []
    for _ in range(total_steps):
        lrs.append(lr_scheduler.get_last_lr())
        lr_scheduler.step()
    plt.plot(lrs)
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Learning rate")
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{title}.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./figures")
    args = parser.parse_args()

    sns.set_style("whitegrid")
    net = torch.nn.Linear(10, 10)
    total_steps = int(1e6)
    lr = 0.0001
    eta_min = 0.000001
    warmup_steps = 50000
    
    # cosine
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    sched_kwargs = {"eta_min": eta_min, "T_max": total_steps}
    lr_scheduler = make_lr_scheduler(optimizer, kind="cosine", sched_kwargs=sched_kwargs)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))
    lr_scheduler = SequentialLR(optimizer, [warmup, lr_scheduler], milestones=[warmup_steps])
    visualize_lr_scheduler(lr_scheduler, total_steps=total_steps, title="cosine", save_dir=args.save_dir)

    # cosine
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    sched_kwargs = {"eta_min": eta_min, "T_max": total_steps / 5}
    lr_scheduler = make_lr_scheduler(optimizer, kind="cosine", sched_kwargs=sched_kwargs)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))
    lr_scheduler = SequentialLR(optimizer, [warmup, lr_scheduler], milestones=[warmup_steps])
    visualize_lr_scheduler(lr_scheduler, total_steps=total_steps, title="cosine_2", save_dir=args.save_dir)

    # cosine_restart
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    sched_kwargs = {"eta_min": eta_min, "T_0": int(total_steps / 5)}
    lr_scheduler = make_lr_scheduler(optimizer, kind="cosine_restart", sched_kwargs=sched_kwargs)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))
    lr_scheduler = SequentialLR(optimizer, [warmup, lr_scheduler], milestones=[warmup_steps])
    visualize_lr_scheduler(lr_scheduler, total_steps=total_steps, title="cosine_restart", save_dir=args.save_dir)

    # cosine_restart
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    sched_kwargs = {"eta_min": eta_min, "T_0": int(total_steps / 5), "T_mult": 2}
    lr_scheduler = make_lr_scheduler(optimizer, kind="cosine_restart", sched_kwargs=sched_kwargs)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))
    lr_scheduler = SequentialLR(optimizer, [warmup, lr_scheduler], milestones=[warmup_steps])
    visualize_lr_scheduler(lr_scheduler, total_steps=total_steps, title="cosine_restart_2", save_dir=args.save_dir)

    # cyclic
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    sched_kwargs = {"base_lr": lr, "max_lr": 0.001, "step_size_up": 1e5}
    lr_scheduler = make_lr_scheduler(optimizer, kind="cyclic", sched_kwargs=sched_kwargs)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))
    lr_scheduler = SequentialLR(optimizer, [warmup, lr_scheduler], milestones=[warmup_steps])
    visualize_lr_scheduler(lr_scheduler, total_steps=total_steps, title="cyclic", save_dir=args.save_dir)

    # step
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    sched_kwargs = {"gamma": 0.1, "step_size": 2e5}
    lr_scheduler = make_lr_scheduler(optimizer, kind="step", sched_kwargs=sched_kwargs)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))
    lr_scheduler = SequentialLR(optimizer, [warmup, lr_scheduler], milestones=[warmup_steps])
    visualize_lr_scheduler(lr_scheduler, total_steps=total_steps, title="step", save_dir=args.save_dir)

    # exponential
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    sched_kwargs = {"gamma": eta_min}
    lr_scheduler = make_lr_scheduler(optimizer, kind="exp", sched_kwargs=sched_kwargs)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))
    lr_scheduler = SequentialLR(optimizer, [warmup, lr_scheduler], milestones=[warmup_steps])
    visualize_lr_scheduler(lr_scheduler, total_steps=total_steps, title="exponential", save_dir=args.save_dir)
