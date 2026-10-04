import torch
from ..training.run import write_json


def optimize(models, loss_fn, steps, lr=1e-3, lbfgs_steps=0, lbfgs_max_iter=5,
             log_path=None, stage="adam", accumulation=1, log_interval=10):
    """Finite-loss checks; deterministic fixed closure for the supplied batch."""
    params = [p for m in models for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    history = []
    optimizer.zero_grad(set_to_none=True)

    def checked(step, phase, backward=True):
        terms = loss_fn(step)
        loss = sum(terms.values())
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite loss in {phase} step {step}")
        if backward:
            (loss/accumulation if phase == stage else loss).backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError("Non-finite parameter gradients")
        row = dict(step=step, stage=phase, total=float(loss.detach()),
                   terms={k: float(v.detach()) for k, v in terms.items()})
        return loss, row

    for step in range(1, steps+1):
        _, row = checked(step, stage)
        if step % accumulation == 0 or step == steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        history.append(row)
        if step == 1 or step % log_interval == 0 or step == steps:
            print(f"{stage} {step}/{steps}: {row['total']:.6g}", flush=True)
            if log_path:
                write_json(log_path, history)
    if lbfgs_steps:
        solver = torch.optim.LBFGS(params, lr=1., max_iter=lbfgs_max_iter,
                    history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-7)
        for step in range(1, lbfgs_steps+1):
            def closure():
                solver.zero_grad(set_to_none=True)
                value, _ = checked(steps, "lbfgs")
                return value
            solver.step(closure)
            _, row = checked(steps, "lbfgs", backward=False)
            history.append(row)
    if log_path:
        write_json(log_path, history)
    return history
